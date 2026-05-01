# Databricks notebook source
# MAGIC %md
# MAGIC # Volve Drilling Advisor
# MAGIC ## Phase 2: Tool-Calling Advisory Agent Loop
# MAGIC
# MAGIC Tool-calling agent that monitors real Volve F-15 drilling parameters
# MAGIC window by window — simulating real-time rig floor conditions.
# MAGIC
# MAGIC Claude autonomously decides:
# MAGIC - When to query formation context from offset_well_crew Silver tables
# MAGIC - When to check drillability forecasts vs actual ROP
# MAGIC - When to escalate vs monitor
# MAGIC - What specific action to recommend to the driller
# MAGIC
# MAGIC **Cost controls:**
# MAGIC - Max 15 windows processed
# MAGIC - Max 8 tool calls per window
# MAGIC - Running token usage printed after each window
# MAGIC - Independent conversation per window — no history accumulation

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

# Cost controls
MAX_WINDOWS        = 15   # maximum windows to process
MAX_TOOLS_PER_WIN  = 8    # maximum tool calls per window
WINDOW_SIZE        = 5    # rows per window (25m)
STEP_SIZE          = 10   # advance 10 rows (~50m) between windows
                          # covers key depth intervals without redundancy

# Key depth intervals to focus on
FOCUS_DEPTHS = [3305, 3350, 3400, 3500, 3600, 3650, 3700,
                3750, 3800, 3850, 3900, 3950, 4000, 4050, 4085]

print("Configuration loaded.")
print(f"Max windows: {MAX_WINDOWS}")
print(f"Max tool calls per window: {MAX_TOOLS_PER_WIN}")
print(f"Step size: {STEP_SIZE} rows ({STEP_SIZE * 5}m)")

# COMMAND ----------
# MAGIC %md ### Step 3: Re-define streaming simulation + tools

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime
import json, time, pandas as pd
import anthropic

client = anthropic.Anthropic()

# ── Load data ─────────────────────────────────────────────────────────────────
df_pd = spark.table("drilling_advisor.bronze_drilling_parameters") \
    .orderBy("Depth").toPandas()

print(f"Loaded {len(df_pd)} rows from bronze_drilling_parameters")
print(f"Depth range: {df_pd['Depth'].min()}m — {df_pd['Depth'].max()}m")

# ── Streaming window function ──────────────────────────────────────────────────
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

# ── Tool functions ─────────────────────────────────────────────────────────────
def get_formation_context(depth_m):
    """Query offset_well_crew Silver tables for formation context at depth."""
    try:
        # Formation tops
        tops = spark.table("offset_well_crew.silver_formation_tops") \
            .select("formation", "picked_depth_m", "offset_avg_depth_m",
                    "depth_shift_m", "severity").collect()
        tops_list = [row.asDict() for row in tops]

        # Nearest reservoir flag
        flags = spark.table("offset_well_crew.silver_reservoir_flags") \
            .filter(
                (F.col("depth_from_m") <= depth_m + 50) &
                (F.col("depth_to_m")   >= depth_m - 50)
            ) \
            .select("depth_from_m", "depth_to_m", "flag_type", "severity",
                    "current_well_character", "recommendation") \
            .orderBy(F.abs(F.col("depth_from_m") - depth_m)) \
            .limit(3).collect()
        flags_list = [row.asDict() for row in flags]

        # Determine which formation we're in
        formation_position = "ABOVE_DRAUPNE"
        for top in tops_list:
            if top["formation"] == "DRAUPNE" and depth_m >= top["picked_depth_m"]:
                formation_position = "IN_DRAUPNE"
            if top["formation"] == "HUGIN_TOP" and depth_m >= top["picked_depth_m"]:
                formation_position = "IN_HUGIN_RESERVOIR"
            if top["formation"] == "HUGIN_BASE" and depth_m >= top["picked_depth_m"]:
                formation_position = "BELOW_HUGIN"

        return {
            "depth_m":            depth_m,
            "formation_position": formation_position,
            "formation_tops":     tops_list,
            "nearby_flags":       flags_list,
        }
    except Exception as e:
        return {"error": str(e), "depth_m": depth_m}

def get_drillability_forecast(depth_m):
    """Get expected drillability at current depth from offset well analogs."""
    try:
        rows = spark.table("offset_well_crew.silver_drillability_forecast") \
            .filter(
                (F.col("depth_from_m") <= depth_m + 50) &
                (F.col("depth_to_m")   >= depth_m - 50)
            ) \
            .select("depth_from_m", "depth_to_m", "expected_drillability", "basis") \
            .orderBy(F.abs(F.col("depth_from_m") - depth_m)) \
            .limit(2).collect()
        return {
            "depth_m":  depth_m,
            "forecast": [row.asDict() for row in rows],
            "available": len(rows) > 0
        }
    except Exception as e:
        return {"error": str(e), "depth_m": depth_m}

def check_rop_efficiency(window_index, current_depth):
    """Compare current ROP against recent trend and offset analog."""
    try:
        # Get last 5 windows for trend
        start  = max(0, window_index - 4)
        recent = df_pd.iloc[start * STEP_SIZE : window_index * STEP_SIZE + WINDOW_SIZE]
        trend_rop = round(float(recent["ROP_mhr"].mean()), 2)
        current_rop = round(float(
            df_pd.iloc[window_index * STEP_SIZE :
                       window_index * STEP_SIZE + WINDOW_SIZE]["ROP_mhr"].mean()
        ), 2)
        pct_change = round((current_rop - trend_rop) / trend_rop * 100, 1) \
                     if trend_rop > 0 else 0.0
        return {
            "current_depth":  current_depth,
            "current_rop":    current_rop,
            "trend_rop_5win": trend_rop,
            "pct_change":     pct_change,
            "assessment":     "ROP_DROP" if pct_change < -20 else
                              "ROP_INCREASE" if pct_change > 20 else "STABLE",
        }
    except Exception as e:
        return {"error": str(e)}

def check_mse_efficiency(window_data):
    """Assess bit efficiency from MSE proxy."""
    mse = window_data["parameters"]["MSE_proxy_mean"]
    rop = window_data["parameters"]["ROP_mhr_mean"]
    wob = window_data["parameters"]["WOB_mean_N"]

    # MSE thresholds — empirical for this well
    if mse > 50000:
        assessment = "INEFFICIENT — high MSE, bit working hard for low ROP"
    elif mse > 30000:
        assessment = "MODERATE — acceptable but monitor for trend"
    else:
        assessment = "EFFICIENT — good ROP for applied WOB"

    return {
        "MSE_proxy":  mse,
        "ROP_mhr":    rop,
        "WOB_N":      wob,
        "assessment": assessment,
        "recommendation": "Consider reducing WOB or optimizing RPM" if mse > 50000
                          else "Parameters within acceptable range"
    }

def dispatch_tool(name, inp, window_data, window_index):
    dispatch = {
        "get_formation_context":    lambda: get_formation_context(inp.get("depth_m", window_data["current_depth"])),
        "get_drillability_forecast": lambda: get_drillability_forecast(inp.get("depth_m", window_data["current_depth"])),
        "check_rop_efficiency":     lambda: check_rop_efficiency(window_index, window_data["current_depth"]),
        "check_mse_efficiency":     lambda: check_mse_efficiency(window_data),
    }
    return dispatch.get(name, lambda: {"error": f"Unknown tool: {name}"})()

TOOL_SCHEMAS = [
    {"name": "get_formation_context",
     "description": "Get formation context from offset well Silver tables at current depth — formation position (Draupne/Hugin), nearby HC flags, reservoir quality flags.",
     "input_schema": {"type": "object",
                      "properties": {"depth_m": {"type": "number", "description": "Current bit depth in meters"}},
                      "required": ["depth_m"]}},
    {"name": "get_drillability_forecast",
     "description": "Get expected drillability at current depth from offset well analog data — HARD/MODERATE/SOFT with geological basis.",
     "input_schema": {"type": "object",
                      "properties": {"depth_m": {"type": "number"}},
                      "required": ["depth_m"]}},
    {"name": "check_rop_efficiency",
     "description": "Compare current ROP against recent trend — detects ROP drops or improvements vs last 5 windows.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "check_mse_efficiency",
     "description": "Assess bit efficiency using MSE proxy — determines if bit is working efficiently or overworked for current ROP.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
]

print("Tools and streaming simulation ready.")

# COMMAND ----------
# MAGIC %md ### Step 4: Advisory Agent — per window
# MAGIC
# MAGIC Each window is an independent conversation.
# MAGIC No history accumulation between windows.
# MAGIC Claude decides which tools to call based on what the data shows.

# COMMAND ----------

def run_advisory_agent(window_data, window_index):
    """
    Tool-calling advisory agent for a single depth window.
    Independent conversation — no cross-window history.
    Returns advisory + token usage.
    """
    depth = window_data["current_depth"]
    params = window_data["parameters"]

    goal = f"""You are a real-time drilling advisor monitoring well 15/9-F-15 in the Volve field.

Current bit depth: {depth}m (window {window_index})
Depth interval: {window_data['depth_from']}m – {window_data['depth_to']}m

Current drilling parameters:
{json.dumps(params, indent=2)}

Key context:
- Hugin reservoir entry at ~3,350m (offset well analog)
- CRITICAL HC potential flagged at 3,350m and 3,700m by offset well crew
- Best confirmed reservoir: 3,800–3,900m
- CRITICAL anomaly at 4,000–4,050m (shale where offsets show HC sand)

Use your tools to investigate the current situation and provide a specific,
actionable drilling advisory. Be concise — the driller needs clear guidance.
Focus on: what is happening, why, and what to do about it."""

    system = """You are an expert drilling advisor with 20 years of North Sea experience.
You receive real-time drilling data window by window and provide actionable recommendations.
Use tools to get formation context and assess efficiency before advising.
Keep advisories concise and specific — depth-referenced, parameter-specific recommendations."""

    messages = [{"role": "user", "content": goal}]
    tool_log = []
    total_input_tokens  = 0
    total_output_tokens = 0
    iteration = 0

    while iteration < MAX_TOOLS_PER_WIN:
        iteration += 1
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages
        )

        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            final = "".join(b.text for b in response.content if hasattr(b, "text"))
            return {
                "window_index":       window_index,
                "current_depth":      depth,
                "depth_from":         window_data["depth_from"],
                "depth_to":           window_data["depth_to"],
                "parameters":         params,
                "advisory":           final,
                "tool_calls":         len(tool_log),
                "tool_log":           tool_log,
                "input_tokens":       total_input_tokens,
                "output_tokens":      total_output_tokens,
                "total_tokens":       total_input_tokens + total_output_tokens,
                "estimated_cost_usd": round(
                    (total_input_tokens / 1_000_000 * 3.0) +
                    (total_output_tokens / 1_000_000 * 15.0), 5
                )
            }

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input,
                                          window_data, window_index)
                    tool_log.append({"tool": block.name, "input": block.input})
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": json.dumps(result, default=str)
                    })
            messages.append({"role": "user", "content": tool_results})

    return {"error": "max_tools_reached", "window_index": window_index}

print("Advisory agent defined.")

# COMMAND ----------
# MAGIC %md ### Step 5: Run agent across selected depth windows

# COMMAND ----------

# Select windows at focus depths
selected_windows = []
for focus_depth in FOCUS_DEPTHS:
    # Find closest window index for this depth
    depth_diffs = abs(df_pd["Depth"] - focus_depth)
    closest_row = depth_diffs.idxmin()
    window_idx  = max(0, closest_row - WINDOW_SIZE // 2)
    window_idx  = window_idx // STEP_SIZE
    if window_idx not in selected_windows:
        selected_windows.append(window_idx)

selected_windows = sorted(set(selected_windows))[:MAX_WINDOWS]
print(f"Selected {len(selected_windows)} windows: {selected_windows}")

# Run agent
all_advisories = []
total_cost     = 0.0
total_tokens   = 0

print(f"\n{'='*60}")
print(f"VOLVE DRILLING ADVISOR — Well 15/9-F-15")
print(f"{'='*60}\n")

for i, win_idx in enumerate(selected_windows):
    window_data = get_drilling_window(win_idx)
    if window_data is None:
        continue

    print(f"[{i+1:02d}/{len(selected_windows)}] Depth {window_data['current_depth']}m — analyzing...")

    result = run_advisory_agent(window_data, win_idx)

    if "error" not in result:
        all_advisories.append(result)
        total_cost   += result["estimated_cost_usd"]
        total_tokens += result["total_tokens"]

        print(f"  Tools called: {result['tool_calls']} | "
              f"Tokens: {result['total_tokens']:,} | "
              f"Cost: ${result['estimated_cost_usd']:.4f} | "
              f"Running total: ${total_cost:.4f}")
        print(f"  Advisory preview: {result['advisory'][:120]}...")
        print()
    else:
        print(f"  Error: {result['error']}")

print(f"\n{'='*60}")
print(f"COMPLETE — {len(all_advisories)} advisories generated")
print(f"Total tokens: {total_tokens:,}")
print(f"Total estimated cost: ${total_cost:.4f}")
print(f"{'='*60}")

# COMMAND ----------
# MAGIC %md ### Step 6: Print full advisories

# COMMAND ----------

for adv in all_advisories:
    print(f"\n{'='*60}")
    print(f"DEPTH: {adv['current_depth']}m | Tools: {adv['tool_calls']} | Tokens: {adv['total_tokens']:,}")
    print(f"{'='*60}")
    print(adv["advisory"])

# COMMAND ----------
# MAGIC %md ### Step 7: Write Silver Delta table

# COMMAND ----------

if all_advisories:
    advisory_rows = []
    for adv in all_advisories:
        advisory_rows.append({
            "run_timestamp":    datetime.now().isoformat(),
            "window_index":     adv["window_index"],
            "current_depth_m":  adv["current_depth"],
            "depth_from_m":     adv["depth_from"],
            "depth_to_m":       adv["depth_to"],
            "WOB_mean_N":       adv["parameters"]["WOB_mean_N"],
            "RPM_mean":         adv["parameters"]["RPM_mean"],
            "ROP_mhr_mean":     adv["parameters"]["ROP_mhr_mean"],
            "MSE_proxy_mean":   adv["parameters"]["MSE_proxy_mean"],
            "formation_class":  adv["parameters"]["formation_class"],
            "reservoir_quality": adv["parameters"]["reservoir_quality"],
            "ROP_drop_flag":    adv["parameters"]["ROP_drop_flag"],
            "advisory":         adv["advisory"],
            "tool_calls":       adv["tool_calls"],
            "total_tokens":     adv["total_tokens"],
            "estimated_cost_usd": adv["estimated_cost_usd"],
            "tool_log_json":    json.dumps(adv["tool_log"], default=str),
        })

    df_silver = spark.createDataFrame(pd.DataFrame(advisory_rows))
    (df_silver.write.format("delta").mode("overwrite")
        .saveAsTable("drilling_advisor.silver_advisories"))

    print(f"Written {len(advisory_rows)} advisories to drilling_advisor.silver_advisories")

    print("\n=== Advisory summary by depth ===")
    spark.table("drilling_advisor.silver_advisories") \
        .select("current_depth_m", "formation_class", "ROP_drop_flag",
                "tool_calls", "total_tokens", "estimated_cost_usd") \
        .orderBy("current_depth_m") \
        .show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 2 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `drilling_advisor.bronze_drilling_parameters` | Raw F-15 drilling data with derived parameters |
# MAGIC | `drilling_advisor.silver_advisories` | AI advisory per depth window with token usage |
# MAGIC
# MAGIC **Next:** Phase 3 — Gold Report + README + LinkedIn Card
