# Databricks notebook source
# MAGIC %md
# MAGIC # Volve Drilling Advisor
# MAGIC ## Phase 1: Data Ingestion + Streaming Simulation Setup
# MAGIC
# MAGIC Ingests real Volve field drilling data for well 15/9-F-15 and sets up
# MAGIC the streaming simulation infrastructure — replaying depth windows
# MAGIC sequentially to simulate real-time rig floor conditions.
# MAGIC
# MAGIC **Well:** 15/9-F-15, Volve Field, Norwegian Continental Shelf
# MAGIC **Depth range:** 3,305m – 4,085m (153 rows at 5m intervals)
# MAGIC **Columns:** Depth, WOB, SURF_RPM, ROP_AVG, PHIF, VSH, SW, KLOGH
# MAGIC
# MAGIC **Derived parameters:**
# MAGIC - `ROP_mhr` — ROP converted from m/s to m/hr
# MAGIC - `MSE_proxy` — Mechanical Specific Energy proxy (WOB × RPM / ROP)
# MAGIC - `formation_class` — classified from VSH and PHIF
# MAGIC
# MAGIC **Cross-project connection:**
# MAGIC Depth range overlaps with offset_well_crew Silver tables (3,350–3,950m)
# MAGIC Formation context pulled from silver_formation_tops and silver_reservoir_flags

# COMMAND ----------
# MAGIC %md ### Step 1: Create database

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS drilling_advisor")
spark.sql("USE drilling_advisor")
print("Database ready: drilling_advisor")

# COMMAND ----------
# MAGIC %md ### Step 2: Upload CSV to Volume
# MAGIC
# MAGIC Upload `ROP_data.csv` to:
# MAGIC `/Volumes/workspace/offset_well_crew/volve_data/ROP_data.csv`
# MAGIC
# MAGIC Use the same Volume as offset_well_crew — keeps all Volve data together.

# COMMAND ----------
# MAGIC %md ### Step 3: Load and inspect raw data

# COMMAND ----------

from pyspark.sql.functions import (
    col, round as spark_round, when, lit
)
from pyspark.sql.types import DoubleType
import pandas as pd

# Load raw CSV
df_raw = spark.read.csv(
    "/Volumes/workspace/offset_well_crew/volve_data/ROP_data.csv",
    header=True,
    inferSchema=True
)

print(f"Raw rows: {df_raw.count()}")
print(f"Columns: {df_raw.columns}")
df_raw.show(5)

# COMMAND ----------
# MAGIC %md ### Step 4: Engineer derived parameters

# COMMAND ----------

# Convert ROP from m/s to m/hr
df = df_raw.withColumn("ROP_mhr", spark_round(col("ROP_AVG") * 3600, 2))

# MSE proxy — Mechanical Specific Energy
# Higher MSE = bit working harder for less penetration = inefficiency
# Formula: WOB × SURF_RPM / ROP_mhr
# Units: N × rpm / (m/hr) — relative proxy, not absolute MSE
df = df.withColumn(
    "MSE_proxy",
    spark_round(
        when(col("ROP_mhr") > 0, col("WOB") * col("SURF_RPM") / col("ROP_mhr"))
        .otherwise(None),
        2
    )
)

# Simulated Torque — derived from WOB and RPM
# Torque_est (N.m) = WOB × 0.3 / RPM (empirical proxy for PDC bit)
df = df.withColumn(
    "Torque_est",
    spark_round(
        when(col("SURF_RPM") > 0, col("WOB") * 0.3 / col("SURF_RPM"))
        .otherwise(None),
        2
    )
)

# Formation classification from petrophysical parameters
# VSH < 0.1 = clean sand, VSH > 0.3 = shale, between = transition
# PHIF > 0.08 = porous, < 0.05 = tight
df = df.withColumn(
    "formation_class",
    when(col("VSH") < 0.1, "CLEAN_SAND")
    .when(col("VSH") > 0.3, "SHALE")
    .otherwise("TRANSITION")
)

# Reservoir quality flag from petrophysical parameters
df = df.withColumn(
    "reservoir_quality",
    when(
        (col("VSH") < 0.1) & (col("PHIF") > 0.08) & (col("SW") < 0.95),
        "GOOD"
    ).when(
        (col("VSH") < 0.1) & (col("PHIF") > 0.05),
        "MODERATE"
    ).otherwise("POOR")
)

# ROP efficiency flag — flag sudden ROP drops
# Compute rolling average comparison in pandas
df_pd = df.orderBy("Depth").toPandas()
df_pd["ROP_rolling_avg"] = df_pd["ROP_mhr"].rolling(window=5, min_periods=1).mean()
df_pd["ROP_drop_flag"] = (
    df_pd["ROP_mhr"] < df_pd["ROP_rolling_avg"] * 0.7
)  # ROP dropped >30% vs rolling average

# Convert back to Spark
df_enriched = spark.createDataFrame(df_pd)

print(f"Enriched rows: {df_enriched.count()}")
print(f"Columns: {df_enriched.columns}")
df_enriched.show(5)

# COMMAND ----------
# MAGIC %md ### Step 5: Write Bronze Delta table

# COMMAND ----------

(df_enriched
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("drilling_advisor.bronze_drilling_parameters")
)

print("Bronze table written: drilling_advisor.bronze_drilling_parameters")

# Validate
print("\n=== Formation class distribution ===")
df_enriched.groupBy("formation_class", "reservoir_quality").count() \
    .orderBy("formation_class").show()

print("\n=== Depth range and key stats ===")
df_enriched.select(
    "Depth", "WOB", "SURF_RPM", "ROP_mhr", "MSE_proxy", "formation_class"
).orderBy("Depth").show(10)

# COMMAND ----------
# MAGIC %md ### Step 6: Set up streaming simulation infrastructure
# MAGIC
# MAGIC Streaming simulation: replay depth windows sequentially
# MAGIC Each window = 5 consecutive depth readings (25m interval)
# MAGIC Window advances one row at a time — simulates bit moving through formation

# COMMAND ----------

WINDOW_SIZE = 5   # rows per window — 25m
STEP_SIZE   = 1   # rows to advance per step

def get_drilling_window(window_index, df_pd):
    """
    Return a single depth window of drilling data.
    Simulates real-time sensor stream at current bit depth.
    """
    start = window_index * STEP_SIZE
    end   = start + WINDOW_SIZE

    if start >= len(df_pd):
        return None  # end of well

    window = df_pd.iloc[start:end].copy()
    current_depth = float(window["Depth"].iloc[-1])

    return {
        "window_index":  window_index,
        "current_depth": current_depth,
        "depth_from":    float(window["Depth"].iloc[0]),
        "depth_to":      current_depth,
        "rows":          len(window),
        "parameters": {
            "WOB_mean_N":       round(window["WOB"].mean(), 2),
            "WOB_std_N":        round(window["WOB"].std(), 2),
            "RPM_mean":         round(window["SURF_RPM"].mean(), 3),
            "ROP_mhr_mean":     round(window["ROP_mhr"].mean(), 2),
            "ROP_mhr_min":      round(window["ROP_mhr"].min(), 2),
            "MSE_proxy_mean":   round(window["MSE_proxy"].mean(), 2),
            "MSE_proxy_max":    round(window["MSE_proxy"].max(), 2),
            "PHIF_mean":        round(window["PHIF"].mean(), 4),
            "VSH_mean":         round(window["VSH"].mean(), 4),
            "SW_mean":          round(window["SW"].mean(), 4),
            "Torque_est_mean":  round(window["Torque_est"].mean(), 2),
            "ROP_drop_flag":    bool(window["ROP_drop_flag"].any()),
            "formation_class":  window["formation_class"].mode()[0],
            "reservoir_quality": window["reservoir_quality"].mode()[0],
        }
    }

# Test the streaming simulation
df_pd_sorted = spark.table("drilling_advisor.bronze_drilling_parameters") \
    .orderBy("Depth").toPandas()

total_windows = len(df_pd_sorted) - WINDOW_SIZE + 1
print(f"Total windows available: {total_windows}")
print(f"Depth range: {df_pd_sorted['Depth'].min()}m — {df_pd_sorted['Depth'].max()}m")

# Preview first 3 windows
for i in range(3):
    window = get_drilling_window(i, df_pd_sorted)
    print(f"\n=== Window {i} | Depth {window['depth_from']}–{window['depth_to']}m ===")
    for k, v in window["parameters"].items():
        print(f"  {k:25}: {v}")

# COMMAND ----------
# MAGIC %md ### Step 7: Verify cross-project connection to offset_well_crew

# COMMAND ----------

# Confirm offset_well_crew Silver tables are accessible
# The advisor will query these for formation context at current depth

print("=== Checking offset_well_crew Silver tables ===")

# Formation tops — Draupne and Hugin picks
print("\n--- silver_formation_tops ---")
spark.table("offset_well_crew.silver_formation_tops") \
    .select("formation", "picked_depth_m", "offset_avg_depth_m", "depth_shift_m", "severity") \
    .show()

# Reservoir flags — HC potential and drillability at depth
print("\n--- silver_reservoir_flags (depth range overlap) ---")
spark.table("offset_well_crew.silver_reservoir_flags") \
    .filter(
        (col("depth_from_m") >= 3300) & (col("depth_to_m") <= 4100)
    ) \
    .select("depth_from_m", "depth_to_m", "flag_type", "severity",
            "current_well_character", "recommendation") \
    .orderBy("depth_from_m") \
    .show(truncate=70)

# Drillability forecast — expected conditions at each depth
print("\n--- silver_drillability_forecast ---")
spark.table("offset_well_crew.silver_drillability_forecast") \
    .filter(
        (col("depth_from_m") >= 3300) & (col("depth_to_m") <= 4100)
    ) \
    .select("depth_from_m", "depth_to_m", "expected_drillability", "basis") \
    .orderBy("depth_from_m") \
    .show(truncate=70)

# COMMAND ----------
# MAGIC %md ### Step 8: Write well metadata table

# COMMAND ----------

from pyspark.sql import Row

well_metadata = [Row(
    well_name       = "15/9-F-15",
    field           = "Volve",
    location        = "Norwegian Continental Shelf, Block 15/9",
    depth_from_m    = 3305.0,
    depth_to_m      = 4085.0,
    total_rows      = 151,
    depth_interval_m= 5.0,
    source_file     = "ROP_data.csv",
    curves          = "Depth,WOB,SURF_RPM,ROP_AVG,PHIF,VSH,SW,KLOGH",
    derived_curves  = "ROP_mhr,MSE_proxy,Torque_est,formation_class,reservoir_quality,ROP_drop_flag",
    cross_project   = "offset_well_crew.silver_formation_tops,offset_well_crew.silver_reservoir_flags,offset_well_crew.silver_drillability_forecast"
)]

df_meta = spark.createDataFrame(well_metadata)
(df_meta.write.format("delta").mode("overwrite")
    .saveAsTable("drilling_advisor.well_metadata"))

print("Well metadata written: drilling_advisor.well_metadata")
df_meta.show(truncate=60)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 1 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `drilling_advisor.bronze_drilling_parameters` | 153 rows of real Volve F-15 drilling data with derived MSE, Torque, formation class |
# MAGIC | `drilling_advisor.well_metadata` | Well metadata including cross-project reference |
# MAGIC
# MAGIC **Streaming simulation ready:**
# MAGIC - 149 windows available (5-row / 25m windows, 1-row step)
# MAGIC - Depth 3,305m – 4,085m
# MAGIC - Cross-project connection to offset_well_crew Silver tables verified
# MAGIC
# MAGIC **Next:** Phase 2 — Tool-Calling Advisory Agent Loop
