# Databricks notebook source
# MAGIC %md
# MAGIC # Volve Drilling Advisor
# MAGIC ## Phase 3: Rescan + Gold Report Synthesis
# MAGIC
# MAGIC Two goals in one phase — efficient by design:
# MAGIC
# MAGIC **Goal 1:** Re-run the 3 ceiling-hit windows (3,325m, 3,645m, 3,895m)
# MAGIC with formation context pre-loaded in the prompt — saves 1-2 tool calls
# MAGIC per window, allowing deeper investigation within a 12-call ceiling.
# MAGIC
# MAGIC **Goal 2:** Synthesize all 13 advisories into a Gold well report —
# MAGIC a complete AI-generated drilling intelligence document for well 15/9-F-15.

# COMMAND ----------
# MAGIC %md ### Step 1: Install Anthropic SDK

# COMMAND ----------

# MAGIC %pip install anthropic
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ### Step 2: Configuration

# COMMAND ----------

import os
os.environ["ANTHROPIC_API_KEY"] = "<YOUR_API_KEY_HERE>"

REPORT_PATH        = "/Volumes/workspace/offset_well_crew/volve_data/drilling_advisory_report.md"
MAX_TOOLS_RESCAN   = 12   # higher ceiling for rescanned windows
WINDOW_SIZE        = 5
STEP_SIZE          = 10
RESCAN_DEPTHS      = [3325.0, 3645.0, 3895.0]  # ceiling-hit windows

print("Configuration loaded.")

# COMMAND ----------
# MAGIC %md ### Step 3: Re-define tools and streaming simulation

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime
import json, pandas as pd
import anthropic

client = anthropic.Anthropic()

df_pd = spark.table("drilling_advisor.bronze_drilling_parameters") \
    .orderBy("Depth").toPandas()

def get_drilling_window(window_index):
    start = window_index * STEP_SIZE
    end   = start + WINDOW_SIZE
    if start >= len(df_pd): return None
    window = df_pd.iloc[start:min(end, len(df_pd))].copy()
    current_depth = float(window["Depth"].iloc[-1])
    return {
        "window_index":  window_index,
        "current_depth": current_depth,
        "depth_from":    float(window["Depth"].iloc[0]),
        "depth_to":      current_depth,
        "parameters": {
            "WOB_mean_N":        round(float(window["WOB"].mean()), 2),
            "WOB_std_N":         round(float(window["WOB"].std()), 2),
            "RPM_mean":          round(float(window["SURF_RPM"].mean()), 3),
            "ROP_mhr_mean":      round(float(window["ROP_mhr"].mean()), 2),
            "ROP_mhr_min":       round(float(window["ROP_mhr"].min()), 2),
            "MSE_proxy_mean":    round(float(window["MSE_proxy"].mean()), 2),
            "MSE_proxy_max":     round(float(window["MSE_proxy"].max()), 2),
            "PHIF_mean":         round(float(window["PHIF"].mean()), 4),
            "VSH_mean":          round(float(window["VSH"].mean()), 4),
            "SW_mean":           round(float(window["SW"].mean()), 4),
            "Torque_est_mean":   round(float(window["Torque_est"].mean()), 2),
            "ROP_drop_flag":     bool(window["ROP_drop_flag"].any()),
            "formation_class":   str(window["formation_class"].mode()[0]),
            "reservoir_quality": str(window["reservoir_quality"].mode()[0]),
        }
    }

def get_formation_context(depth_m):
    try:
        tops = spark.table("offset_well_crew.silver_formation_tops") \
            .select("formation","picked_depth_m","offset_avg_depth_m",
                    "depth_shift_m","severity").collect()
        tops_list = [r.asDict() for r in tops]
        flags = spark.table("offset_well_crew.silver_reservoir_flags") \
            .filter(
                (F.col("depth_from_m") <= depth_m+50) &
                (F.col("depth_to_m")   >= depth_m-50)
            ).select("depth_from_m","depth_to_m","flag_type","severity",
                     "current_well_character","recommendation") \
            .orderBy(F.abs(F.col("depth_from_m")-depth_m)).limit(3).collect()
        flags_list = [r.asDict() for r in flags]
        position = "ABOVE_DRAUPNE"
        for t in tops_list:
            if t["formation"]=="DRAUPNE" and depth_m>=t["picked_depth_m"]:
                position="IN_DRAUPNE"
            if t["formation"]=="HUGIN_TOP" and depth_m>=t["picked_depth_m"]:
                position="IN_HUGIN_RESERVOIR"
            if t["formation"]=="HUGIN_BASE" and depth_m>=t["picked_depth_m"]:
                position="BELOW_HUGIN"
        return {"depth_m":depth_m,"formation_position":position,
                "formation_tops":tops_list,"nearby_flags":flags_list}
    except Exception as e:
        return {"error":str(e),"depth_m":depth_m}

def get_drillability_forecast(depth_m):
    try:
        rows = spark.table("offset_well_crew.silver_drillability_forecast") \
            .filter(
                (F.col("depth_from_m") <= depth_m+50) &
                (F.col("depth_to_m")   >= depth_m-50)
            ).select("depth_from_m","depth_to_m","expected_drillability","basis") \
            .orderBy(F.abs(F.col("depth_from_m")-depth_m)).limit(2).collect()
        return {"depth_m":depth_m,"forecast":[r.asDict() for r in rows],
                "available":len(rows)>0}
    except Exception as e:
        return {"error":str(e),"depth_m":depth_m}

def check_rop_efficiency(window_index, current_depth):
    try:
        start   = max(0, window_index-4)
        recent  = df_pd.iloc[start*STEP_SIZE : window_index*STEP_SIZE+WINDOW_SIZE]
        trend   = round(float(recent["ROP_mhr"].mean()), 2)
        current = round(float(
            df_pd.iloc[window_index*STEP_SIZE :
                       window_index*STEP_SIZE+WINDOW_SIZE]["ROP_mhr"].mean()), 2)
        pct = round((current-trend)/trend*100, 1) if trend>0 else 0.0
        return {"current_depth":current_depth,"current_rop":current,
                "trend_rop_5win":trend,"pct_change":pct,
                "assessment":"ROP_DROP" if pct<-20 else
                             "ROP_INCREASE" if pct>20 else "STABLE"}
    except Exception as e:
        return {"error":str(e)}

def check_mse_efficiency(window_data):
    mse = window_data["parameters"]["MSE_proxy_mean"]
    assessment = ("INEFFICIENT" if mse>50000 else
                  "MODERATE"    if mse>30000 else "EFFICIENT")
    return {"MSE_proxy":mse,"ROP_mhr":window_data["parameters"]["ROP_mhr_mean"],
            "WOB_N":window_data["parameters"]["WOB_mean_N"],
            "assessment":assessment,
            "recommendation":"Consider reducing WOB or optimizing RPM"
                             if mse>50000 else "Parameters within acceptable range"}

def dispatch_tool(name, inp, window_data, window_index):
    return {
        "get_formation_context":     lambda: get_formation_context(inp.get("depth_m", window_data["current_depth"])),
        "get_drillability_forecast": lambda: get_drillability_forecast(inp.get("depth_m", window_data["current_depth"])),
        "check_rop_efficiency":      lambda: check_rop_efficiency(window_index, window_data["current_depth"]),
        "check_mse_efficiency":      lambda: check_mse_efficiency(window_data),
    }.get(name, lambda: {"error":f"Unknown tool: {name}"})()

TOOL_SCHEMAS = [
    {"name":"get_formation_context",
     "description":"Get formation context from offset well Silver tables — formation position, nearby HC flags. Use when formation context is unclear or conflicting signals exist.",
     "input_schema":{"type":"object","properties":{"depth_m":{"type":"number"}},"required":["depth_m"]}},
    {"name":"get_drillability_forecast",
     "description":"Get expected drillability from offset well analogs — HARD/MODERATE/SOFT.",
     "input_schema":{"type":"object","properties":{"depth_m":{"type":"number"}},"required":["depth_m"]}},
    {"name":"check_rop_efficiency",
     "description":"Compare current ROP vs recent 5-window trend.",
     "input_schema":{"type":"object","properties":{},"required":[]}},
    {"name":"check_mse_efficiency",
     "description":"Assess bit efficiency using MSE proxy.",
     "input_schema":{"type":"object","properties":{},"required":[]}}
]

print("Tools ready.")

# COMMAND ----------
# MAGIC %md ### Step 4: Rescan ceiling-hit windows
# MAGIC
# MAGIC Key efficiency improvement: formation context pre-loaded in the goal prompt.
# MAGIC Claude doesn't need to call get_formation_context first — saves 1-2 tool calls,
# MAGIC leaving more budget for deeper investigation.

# COMMAND ----------

def run_advisory_agent_rescan(window_data, window_index, max_tools=12):
    """
    Advisory agent with pre-loaded formation context.
    Efficiency improvement: formation context in prompt = 1-2 fewer tool calls.
    """
    depth  = window_data["current_depth"]
    params = window_data["parameters"]

    # Pre-load formation context — removes one tool call
    formation_ctx = get_formation_context(depth)
    drillability  = get_drillability_forecast(depth)

    goal = f"""You are a real-time drilling advisor monitoring well 15/9-F-15 in the Volve field.

Current bit depth: {depth}m (window {window_index})
Depth interval: {window_data['depth_from']}m – {window_data['depth_to']}m

Current drilling parameters:
{json.dumps(params, indent=2)}

PRE-LOADED FORMATION CONTEXT (no need to call get_formation_context):
{json.dumps(formation_ctx, indent=2)}

PRE-LOADED DRILLABILITY FORECAST (no need to call get_drillability_forecast):
{json.dumps(drillability, indent=2)}

Key field context:
- Hugin reservoir entry at ~3,350m
- CRITICAL HC potential at 3,350m and 3,700m
- Best confirmed reservoir: 3,800–3,900m
- CRITICAL anomaly at 4,000–4,050m (shale where offsets show HC sand)

Use check_rop_efficiency and check_mse_efficiency to assess drilling performance,
then provide a specific, actionable drilling advisory. Be concise and depth-referenced."""

    system = """You are an expert drilling advisor with 20 years of North Sea experience.
Formation context is pre-loaded — do not call get_formation_context or get_drillability_forecast.
Use check_rop_efficiency and check_mse_efficiency to assess performance.
Provide concise, specific, actionable advisories."""

    messages = [{"role":"user","content":goal}]
    tool_log = []
    total_input = total_output = 0
    iteration = 0

    while iteration < max_tools:
        iteration += 1
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages
        )
        total_input  += response.usage.input_tokens
        total_output += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            final = "".join(b.text for b in response.content if hasattr(b,"text"))
            return {
                "window_index":       window_index,
                "current_depth":      depth,
                "depth_from":         window_data["depth_from"],
                "depth_to":           window_data["depth_to"],
                "parameters":         params,
                "advisory":           final,
                "tool_calls":         len(tool_log),
                "tool_log":           tool_log,
                "input_tokens":       total_input,
                "output_tokens":      total_output,
                "total_tokens":       total_input + total_output,
                "estimated_cost_usd": round(
                    (total_input/1_000_000*3.0) + (total_output/1_000_000*15.0), 5
                ),
                "rescan": True
            }

        if response.stop_reason == "tool_use":
            messages.append({"role":"assistant","content":response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input,
                                          window_data, window_index)
                    tool_log.append({"tool":block.name,"input":block.input})
                    tool_results.append({"type":"tool_result","tool_use_id":block.id,
                                         "content":json.dumps(result,default=str)})
            messages.append({"role":"user","content":tool_results})

    return {"error":"max_tools_reached","window_index":window_index,
            "current_depth":depth, "rescan":True}

# Find window indices for rescan depths
rescan_results = []
total_rescan_cost = 0.0

print("Rescanning ceiling-hit windows with pre-loaded formation context...\n")

for target_depth in RESCAN_DEPTHS:
    depth_diffs = abs(df_pd["Depth"] - target_depth)
    closest_row = depth_diffs.idxmin()
    window_idx  = max(0, closest_row - WINDOW_SIZE // 2) // STEP_SIZE

    window_data = get_drilling_window(window_idx)
    if window_data is None:
        continue

    print(f"Rescanning depth {window_data['current_depth']}m (window {window_idx})...")
    result = run_advisory_agent_rescan(window_data, window_idx, MAX_TOOLS_RESCAN)

    if "error" not in result:
        rescan_results.append(result)
        total_rescan_cost += result["estimated_cost_usd"]
        print(f"  ✅ Tools: {result['tool_calls']} | "
              f"Tokens: {result['total_tokens']:,} | "
              f"Cost: ${result['estimated_cost_usd']:.4f} | "
              f"Running: ${total_rescan_cost:.4f}")
        print(f"  Preview: {result['advisory'][:100]}...\n")
    else:
        print(f"  ❌ Still hitting ceiling at {target_depth}m — skipping\n")

print(f"Rescan complete. {len(rescan_results)} new advisories. Cost: ${total_rescan_cost:.4f}")

# COMMAND ----------
# MAGIC %md ### Step 5: Merge rescan results into Silver table

# COMMAND ----------

# Load existing silver advisories
existing = spark.table("drilling_advisor.silver_advisories").toPandas()
print(f"Existing advisories: {len(existing)}")

# Add rescan results
if rescan_results:
    new_rows = []
    for adv in rescan_results:
        new_rows.append({
            "run_timestamp":      datetime.now().isoformat(),
            "window_index":       adv["window_index"],
            "current_depth_m":    adv["current_depth"],
            "depth_from_m":       adv["depth_from"],
            "depth_to_m":         adv["depth_to"],
            "WOB_mean_N":         adv["parameters"]["WOB_mean_N"],
            "RPM_mean":           adv["parameters"]["RPM_mean"],
            "ROP_mhr_mean":       adv["parameters"]["ROP_mhr_mean"],
            "MSE_proxy_mean":     adv["parameters"]["MSE_proxy_mean"],
            "formation_class":    adv["parameters"]["formation_class"],
            "reservoir_quality":  adv["parameters"]["reservoir_quality"],
            "ROP_drop_flag":      adv["parameters"]["ROP_drop_flag"],
            "advisory":           adv["advisory"],
            "tool_calls":         adv["tool_calls"],
            "total_tokens":       adv["total_tokens"],
            "estimated_cost_usd": adv["estimated_cost_usd"],
            "tool_log_json":      json.dumps(adv["tool_log"], default=str),
        })

    df_new = spark.createDataFrame(pd.DataFrame(new_rows))
    (df_new.write.format("delta").mode("append")
        .saveAsTable("drilling_advisor.silver_advisories"))
    print(f"Added {len(new_rows)} rescan advisories to silver_advisories")

# Reload full silver table
df_silver_full = spark.table("drilling_advisor.silver_advisories") \
    .orderBy("current_depth_m").toPandas()
print(f"Total advisories now: {len(df_silver_full)}")

# COMMAND ----------
# MAGIC %md ### Step 6: Synthesize Gold report

# COMMAND ----------

# Build synthesis context from all advisories
advisories_for_synthesis = []
for _, row in df_silver_full.iterrows():
    advisories_for_synthesis.append({
        "depth_m":         row["current_depth_m"],
        "formation_class": row["formation_class"],
        "ROP_mhr":         row["ROP_mhr_mean"],
        "MSE_proxy":       row["MSE_proxy_mean"],
        "ROP_drop_flag":   row["ROP_drop_flag"],
        "advisory_summary": row["advisory"][:500]  # truncated for token efficiency
    })

synthesis_prompt = f"""You are a senior drilling engineer writing a post-drill intelligence 
report for well 15/9-F-15, Volve Field, Norwegian Continental Shelf.

You have received real-time AI-generated drilling advisories for {len(advisories_for_synthesis)} 
depth windows spanning 3,305m – 4,085m.

Here are the advisories (summarized):
{json.dumps(advisories_for_synthesis, indent=2)}

Write a concise professional POST-DRILL INTELLIGENCE REPORT with these sections:

# Post-Drill Intelligence Report — Well 15/9-F-15
## Volve Field, Block 15/9 | Norwegian Continental Shelf

### 1. Executive Summary (3-4 sentences)

### 2. Formation Encountered
Brief summary of what was drilled — Draupne, Hugin, sub-Hugin.

### 3. Drilling Performance Summary
Key ROP, MSE, and efficiency observations across the well.

### 4. Critical Depth Intervals
Table format: Depth | Finding | Severity | Action Taken/Recommended

### 5. HC Potential Assessment
Summary of HC-bearing intervals identified vs offset well predictions.

### 6. Anomalies & Deviations from Offset Prognosis
Key differences between this well and offset analog behavior.

### 7. Completion Recommendations
Prioritized perforation targets based on advisory findings.

### 8. Lessons for Next Well
3-5 bullet points for the next well in this area.

Be specific, depth-referenced, and concise. This report goes to the drilling superintendent."""

print("Synthesizing Gold report...")
synthesis_response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=3000,
    messages=[{"role":"user","content":synthesis_prompt}]
)

gold_report = synthesis_response.content[0].text
synthesis_tokens = synthesis_response.usage.input_tokens + synthesis_response.usage.output_tokens
synthesis_cost   = round(
    synthesis_response.usage.input_tokens/1_000_000*3.0 +
    synthesis_response.usage.output_tokens/1_000_000*15.0, 5
)
print(f"Gold report generated — Tokens: {synthesis_tokens:,} | Cost: ${synthesis_cost:.4f}")

# COMMAND ----------
# MAGIC %md ### Step 7: Write Gold Delta table

# COMMAND ----------

gold_record = {
    "run_id":              datetime.now().strftime("%Y%m%d_%H%M%S"),
    "run_timestamp":       datetime.now().isoformat(),
    "well_name":           "15/9-F-15",
    "field":               "Volve",
    "depth_from_m":        3305.0,
    "depth_to_m":          4085.0,
    "total_windows":       len(df_silver_full),
    "total_tokens":        int(df_silver_full["total_tokens"].sum()) + synthesis_tokens,
    "total_cost_usd":      round(float(df_silver_full["estimated_cost_usd"].sum()) + synthesis_cost, 4),
    "gold_report":         gold_report,
    "status":              "SUCCESS",
}

df_gold = spark.createDataFrame(pd.DataFrame([gold_record]))
(df_gold.write.format("delta").mode("append")
    .saveAsTable("drilling_advisor.gold_drill_reports"))

print(f"Gold record written: {gold_record['run_id']}")
print(f"Total windows: {gold_record['total_windows']}")
print(f"Total cost: ${gold_record['total_cost_usd']:.4f}")

# COMMAND ----------
# MAGIC %md ### Step 8: Save markdown report to Volume

# COMMAND ----------

md_report = f"""# Post-Drill Intelligence Report — Well 15/9-F-15
**Run ID:** {gold_record['run_id']}
**Timestamp:** {gold_record['run_timestamp']}
**Well:** {gold_record['well_name']} | {gold_record['field']} Field
**Depth:** {gold_record['depth_from_m']}m – {gold_record['depth_to_m']}m
**Windows Analyzed:** {gold_record['total_windows']}
**Total API Cost:** ${gold_record['total_cost_usd']:.4f}

---

{gold_report}
"""

dbutils.fs.put(REPORT_PATH, md_report, overwrite=True)
print(f"Report saved: {REPORT_PATH}")
print("\n=== REPORT PREVIEW (first 40 lines) ===\n")
print("\n".join(gold_report.split("\n")[:40]))

# COMMAND ----------
# MAGIC %md ### Step 9: Validate Gold table

# COMMAND ----------

print("=== Gold Drill Reports ===")
spark.table("drilling_advisor.gold_drill_reports") \
    .select("run_id","well_name","depth_from_m","depth_to_m",
            "total_windows","total_cost_usd","status") \
    .show(truncate=60)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 3 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `drilling_advisor.bronze_drilling_parameters` | Raw F-15 drilling data |
# MAGIC | `drilling_advisor.silver_advisories` | Per-depth advisories (original + rescan) |
# MAGIC | `drilling_advisor.gold_drill_reports` | Synthesized post-drill intelligence report |
# MAGIC
# MAGIC | File | Description |
# MAGIC |------|-------------|
# MAGIC | `drilling_advisory_report.md` | Human-readable post-drill report in Volume |
# MAGIC
# MAGIC **Next:** README + LinkedIn Card
