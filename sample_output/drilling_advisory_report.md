# Data Quality Report
**Run ID:** 20260503_110117
**Timestamp:** 2026-05-03T11:01:17.463810
**Tables Monitored:** 26 | **Tool Calls:** 119 | **Iterations:** 7

---

I now have a complete and thorough picture across all 26 tables. Here is the full assessment.

---

# 📊 Delta Lakehouse Data Quality Assessment — Full Report

**Tables Assessed:** 26 across `ecommerce` (17 tables) and `offset_well_crew` (9 tables)
**Checks Run:** Row Counts · Freshness · Null Rates · Schema · Custom SQL Deep-Dives

---

## ✅ DIMENSION 1 — Row Counts
**Status: ALL CLEAR (26/26 tables NORMAL)**

Every table exactly matches its baseline. No truncations, unexpected deletions, or runaway inserts detected. Row volumes are stable across all bronze, silver, and gold layers.

---

## 🔴 DIMENSION 2 — Freshness
**Status: ALL 26 TABLES FLAGGED AS STALE (>48 hrs threshold)**

All tables exceed the 48-hour freshness SLA. However, context matters greatly — here's the breakdown by severity:

| Tier | Tables | Last Updated | Hours Stale | Notes |
|------|--------|-------------|-------------|-------|
| 🔴 **Critical** | `bronze_orders_stream`, `gold_stream_anomalies` | Apr 14 | ~456 hrs | Stream has been completely idle for **19 days** |
| 🔴 **Critical** | `silver_customers_enriched`, `gold_customer_segments` | Apr 14 | ~462 hrs | Created once via CTAS, **never refreshed** |
| 🟠 **High** | `incident_reports`, `pricing_actions`, `pricing_decisions` | Apr 22 | ~258–259 hrs | Last activity ~11 days ago |
| 🟡 **Moderate** | All remaining ecommerce bronze/silver/gold tables | Apr 27 | ~138 hrs | Last pipeline ran Apr 27 — **~5.8 days stale** |
| 🟡 **Moderate** | All `offset_well_crew` tables | Apr 27–28 | ~121–138 hrs | Created once, not incrementally refreshed |

**Key Freshness Findings:**
- `bronze_orders_stream` contains **only April 14 events** (all 450 rows are from a single day). The stream appears **stopped or disconnected**.
- `silver_customers_enriched` and `gold_customer_segments` were built with `CREATE OR REPLACE TABLE AS SELECT` and have **never been updated** — they are static snapshots.
- The `ecommerce` pipeline's last successful run was **Apr 27 at 16:28** and runs daily, suggesting the pipeline **has not executed in ~5+ days**.

---

## 🟡 DIMENSION 3 — Null Rates
**Status: 2 TABLES WITH NOTABLE NULLS (no automated flags triggered, but findings are significant)**

#### Finding A: `offset_well_crew.bronze_well_logs` — DT Column (Sonic Log)
- **50.51% null rate** on the `DT` (sonic/delta-t) column across 49,305 rows
- Deep-dive reveals this is **well-specific and expected**:

| Well | Role | DT Null % | Notes |
|------|------|-----------|-------|
| `15_9-F-1C` | Current | **100%** | No DT log acquired |
| `15_9-F-11B` | Offset | **100%** | No DT log acquired |
| `15_9-F-1A` | Offset | 0% | DT available |
| `15_9-F-11A` | Offset | 0% | DT available |
| `15_9-F-1B` | Offset | 0% | DT available |

The `well_registry` confirms `has_dt = false` for `15_9-F-1C` and `15_9-F-11B` — this is **documented and intentional**. However, the absence of DT on the **current well** limits sonic-dependent interpretations (velocity modeling, synthetic seismic). ⚠️ Warrants flagging for interpretation teams.

#### Finding B: `ecommerce.gold_customer_segments` — Spend/Order Metrics
- **1.58% null rate** on `total_spend`, `total_orders`, `avg_order_value`
- Affects exactly **3 customers** (IDs 75, 162, 175) across "Loyal" and "Growing" segments
- These are customers with **no matching order history** — a left-join artifact from the segment enrichment query
- Low severity but should be handled with `COALESCE(total_spend, 0)` in the gold layer

#### Finding C: `ecommerce.pipeline_runs` — failed_checks column
- **96.15% null rate** on `failed_checks` — upon investigation this is **expected by design**: the column is only populated when a quality check fails. 19/26 runs succeeded (nulls expected) and only 1 of the 7 failures recorded a check detail. This points to **incomplete failure logging** in failed runs.

#### Finding D: `offset_well_crew.gold_well_reports` — question column
- **25% null** (1 of 4 rows) — the null row is `report_type = 'full_report'`, which has no associated question. This is **by design**.

---

## ✅ DIMENSION 4 — Schema Drift
**Status: ALL CLEAR (26/26 tables NORMAL)**

Zero schema drift detected. No columns added or removed on any table across both projects.

---

## 🔬 DIMENSION 5 — Additional SQL Investigation Findings

### 🏭 Ecommerce Pipeline Health
- **19 successful / 7 failed** runs in history
- All 7 failures occurred in a **concentrated window: Apr 14–15** (early system instability)
- All 6 failures on Apr 15 **stalled at the BRONZE layer** — root cause likely an ingestion/connectivity issue during that period
- The 1 earlier failure (Apr 14) explicitly recorded: *"bronze_orders: row_count (expected >= 5,000, actual 1,000)"* — a quality gate that caught an undersized load
- **Pipeline has been stable since Apr 15** — all recent runs reach GOLD successfully in ~150 seconds avg

### 🛢️ Offset Well Crew — Well Log QC
- **15 QC flags** across all 5 wells:
  - 🔴 **2 CRITICAL flags** affecting 2 wells — requires immediate attention
  - 🟠 **4 MODERATE flags** across 4 wells
  - 🟡 **9 MINOR flags** across 4 wells
- **Formation tops analysis** reveals structural deviations:
  - `HUGIN_BASE` formation is **267m deeper** than offset average (MODERATE severity) — significant structural relief
  - `DRAUPNE` formation is **133m shallower** (MINOR) and `HUGIN_TOP` is **133m deeper** (MINOR)

---

## 📋 PRIORITIZED ACTION ITEMS

### 🔴 P1 — Critical (Immediate Action Required)
| # | Issue | Table(s) | Action |
|---|-------|---------|--------|
| 1 | **Stream completely stopped for 19 days** | `bronze_orders_stream`, `gold_stream_anomalies` | Investigate Structured Streaming job — check for executor failures, checkpoint corruption, or upstream Kafka/event source outage |
| 2 | **2 CRITICAL well log QC flags** | `silver_log_qc_flags` | Review flagged depth intervals immediately — these may compromise interpretation confidence for `15_9-F-1C` |
| 3 | **Silver/Gold customer tables never refreshed** | `silver_customers_enriched`, `gold_customer_segments` | Convert CTAS to scheduled pipeline; data is 19 days stale and does not reflect recent customers or orders |

### 🟠 P2 — High (Address Within 24 Hours)
| # | Issue | Table(s) | Action |
|---|-------|---------|--------|
| 4 | **Ecommerce pipeline not run in ~5.8 days** | All ecommerce bronze/silver/gold | Investigate scheduler — last run was Apr 27; daily cadence has been missed |
| 5 | **Pricing & incident tables ~11 days stale** | `pricing_actions`, `pricing_decisions`, `incident_reports` | Confirm if pricing agent is intentionally paused or has silently failed |
| 6 | **Current well (15_9-F-1C) has no DT log** | `bronze_well_logs` | Notify petrophysicists — sonic-dependent deliverables (synthetics, porosity transforms) are limited |

### 🟡 P3 — Medium (Schedule for Resolution)
| # | Issue | Table(s) | Action |
|---|-------|---------|--------|
| 7 | **3 customers with NULL spend metrics in gold** | `gold_customer_segments` | Add `COALESCE(total_spend, 0)` and `COALESCE(total_orders, 0)` to segment query; investigate why these customers have no orders |
| 8 | **HUGIN_BASE formation 267m deeper than offsets** | `silver_formation_tops` | Flag for geologist review — may indicate fault block or structural complexity |
| 9 | **Incomplete failure logging in pipeline_runs** | `pipeline_runs` | Update pipeline error handler to always write `failed_checks` detail on FAILED status |

---

## 📊 Overall Health Scorecard

| Dimension | ecommerce | offset_well_crew | Overall |
|-----------|-----------|-----------------|---------|
| Row Counts | ✅ PASS | ✅ PASS | ✅ **PASS** |
| Schema | ✅ PASS | ✅ PASS | ✅ **PASS** |
| Null Rates | 🟡 WARN (3 nulls) | 🟡 WARN (DT nulls, documented) | 🟡 **WARN** |
| Freshness | 🔴 FAIL (all stale) | 🔴 FAIL (all stale) | 🔴 **FAIL** |
| **Overall** | 🟠 **DEGRADED** | 🟠 **DEGRADED** | 🟠 **DEGRADED** |

> **Bottom Line:** Both lakehouse domains are structurally sound (schemas intact, row counts stable, minimal nulls) but are suffering from a **widespread freshness failure** — the most urgent concern is the dead streaming pipeline (19 days idle) and the ecommerce daily batch pipeline that has missed its last ~6 scheduled runs. Operational monitoring on pipeline schedulers should be investigated immediately.
