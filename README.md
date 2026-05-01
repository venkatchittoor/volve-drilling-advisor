# volve-drilling-advisor

A constrained tool-calling AI agent that monitors real Volve field drilling parameters window by window — simulating real-time rig floor conditions — and delivers formation-aware drilling advisories by cross-referencing live sensor data against offset well intelligence.

Built with **Claude API** + **Databricks**. Real data from the Norwegian Continental Shelf.

> *"What should the driller do right now — and why?"*
> The advisor answers that question at every depth window, combining real-time drilling mechanics with formation knowledge from offset wells.

---

## Business Impact

> Drilling decisions at depth are time-critical — a missed ROP drop or unrecognized formation change can cost hundreds of thousands of dollars in NPT or missed pay. This advisor reduces decision latency from hours to seconds, synthesizing live drilling parameters and offset well formation knowledge at every depth window. Formation intelligence from one project, drilling mechanics from another — connected in real time.

---

## The Architectural Distinction — Where This Sits on the Spectrum

This project introduces a new pattern to the portfolio:

```
Prompt-based          Constrained              True tool-calling
(prior agents)        tool-calling             (tool-calling-dq-agent)
      |               (this project)                  |
Your code decides   Claude decides             Claude decides
the sequence        which tool + args          which tool + args
                    Queries pre-written        AND writes its own SQL
```

**What makes it tool-calling:**
- Claude decides which tools to call and in what order
- Claude decides when it has enough information to stop
- Your code does not predetermine the investigation sequence

**What makes it constrained:**
- The queries inside each tool are pre-written — Claude cannot go off-script
- The toolkit is narrow and focused (4 tools vs DQ agent's 6 including open-ended SQL)
- Pre-defined data access patterns make it auditable and production-safe

**Why the constraint is a feature, not a limitation:**
In production drilling environments, an agent with arbitrary database access is a liability. Pre-defined tools with known query patterns are safer, auditable, and faster. The constraint reflects engineering judgment — not a capability gap.

---

## Dataset

**Real Volve Field Drilling Data — Well 15/9-F-15**

| Parameter | Detail |
|---|---|
| Field | Volve, Block 15/9, Norwegian Continental Shelf |
| Well | 15/9-F-15 |
| Depth range | 3,305m – 4,085m |
| Interval | 5m sampling — 153 rows |
| Curves | Depth, WOB, SURF_RPM, ROP_AVG, PHIF, VSH, SW, KLOGH |
| Derived | ROP_mhr, MSE_proxy, Torque_est, formation_class, reservoir_quality |

**Cross-project connection:** Formation context is pulled in real time from `offset-well-intelligence-crew` Silver tables — the same Volve field, different data type.

---

## The Toolkit — 4 Constrained Tools

| Tool | Queries from | Purpose |
|---|---|---|
| `get_formation_context` | `offset_well_crew.silver_formation_tops` + `silver_reservoir_flags` | Formation position, HC flags, reservoir quality at current depth |
| `get_drillability_forecast` | `offset_well_crew.silver_drillability_forecast` | Expected HARD/MODERATE/SOFT from offset well analogs |
| `check_rop_efficiency` | `bronze_drilling_parameters` (in-memory) | Current ROP vs 5-window rolling trend |
| `check_mse_efficiency` | Window data — no DB query | Bit efficiency from MSE proxy |

The first two tools reach into `offset-well-intelligence-crew` Silver tables — live cross-project intelligence at every depth window.

---

## Key Domain Concepts

**MSE (Mechanical Specific Energy) proxy:**
```
MSE_proxy = WOB × RPM / ROP_mhr
```
Higher MSE = bit working harder for less penetration = inefficiency flag. Claude uses this to distinguish formation-driven ROP drops from bit-efficiency issues.

**Cross-parameter reasoning at depth:**

| Signal | Alone | Combined with formation context |
|---|---|---|
| ROP drop | Could be bit wear | At 3,350m Hugin entry + HARD drillability = formation-driven, hold WOB |
| ROP drop | Could be bit wear | At 3,800m best reservoir + MODERATE drillability = investigate immediately |
| High MSE | Bit inefficiency | In cemented tight sand = expected, don't chase with WOB |

---

## What the Advisor Produced

**13 depth windows analyzed, 10 successful advisories** across 3,305–4,085m.

**Selected advisory excerpts:**

**3,375m — Hugin entry:**
> *"RPM of 2.6 is anomalously low — target 60–80 RPM. This single change is most likely to recover ROP. Do not chase ROP by adding WOB blindly in this tight cemented matrix — risk of bit balling increases."*

**3,695m — 5m from CRITICAL HC:**
> *"Slow penetration rate at 3,697m. Your next 5 metres are the most important of this well. Have a test plan ready — the signal ahead could be significant."*

**3,795m — 5m from best reservoir:**
> *"You're 5 metres from your best pay. Hold current parameters, sharpen your eyes on LWD, and be ready to call sand entry."*

**3,845m — formation mismatch:**
> *"VSH = 0.83 in a zone with confirmed HC indicators from offsets indicates clay-laminated reservoir, not blanket shale. Do not dismiss this interval."*

**4,005m — CRITICAL anomaly confirmed:**
> *"The reservoir sand predicted at 4,000–4,050m by offsets is not here — you're in shale. The critical decision now is geological, not mechanical."*

See [`sample_output/drilling_advisory_report.md`](sample_output/drilling_advisory_report.md) for the full post-drill intelligence report.

---

## Cross-Project Connection

This project connects directly to [`offset-well-intelligence-crew`](https://github.com/venkatchittoor/offset-well-intelligence-crew):

```
offset-well-intelligence-crew          volve-drilling-advisor
──────────────────────────────         ──────────────────────
silver_formation_tops        ────────→ get_formation_context()
silver_reservoir_flags       ────────→ get_formation_context()
silver_drillability_forecast ────────→ get_drillability_forecast()
```

At 3,700m the advisor says:
> *"Offset well data shows CRITICAL HC potential (RT 3,989 ohm·m). Slow penetration, monitor LWD resistivity and GR in real-time. Flag for formation test consideration."*

That recommendation came from `offset_well_crew.silver_reservoir_flags` — formation knowledge built in one project, deployed in another.

---

## Phases

| Phase | Notebook | Description |
|---|---|---|
| 1 | `Phase1_DataIngestion_StreamingSetup.py` | Ingest F-15 drilling data, derive MSE/Torque, set up streaming simulation, verify cross-project connection |
| 2 | `Phase2_AdvisoryAgentLoop.py` | Tool-calling advisory agent — Claude drives window-by-window investigation |
| 3 | `Phase3_GoldReport.py` | Rescan ceiling-hit windows with pre-loaded context, synthesize Gold post-drill report |

---

## Delta Table Inventory

| Layer | Table | Description |
|---|---|---|
| Bronze | `drilling_advisor.bronze_drilling_parameters` | Raw F-15 drilling data with derived MSE, Torque, formation class |
| Bronze | `drilling_advisor.well_metadata` | Well metadata with cross-project references |
| Silver | `drilling_advisor.silver_advisories` | Per-depth advisories with token usage tracking |
| Gold | `drilling_advisor.gold_drill_reports` | Synthesized post-drill intelligence report |

**Cross-project Silver tables (read-only):**
- `offset_well_crew.silver_formation_tops`
- `offset_well_crew.silver_reservoir_flags`
- `offset_well_crew.silver_drillability_forecast`

---

## Cost Transparency

Token usage is tracked per depth window — printed in real time during execution:

```
[06/13] Depth 3,695m — analyzing...
  Tools called: 4 | Tokens: 17,839 | Cost: $0.1153 | Running total: $0.2833
```

Total Phase 2 run: **88,190 tokens | $0.55** across 13 depth windows.

---

## Tech Stack

| Component | Technology |
|---|---|
| AI reasoning | Claude API (claude-sonnet-4-6) — constrained tool-calling |
| Data platform | Databricks (Serverless) |
| Storage | Delta Lake — Bronze / Silver / Gold |
| Data processing | PySpark + Pandas |
| Source data | Equinor Volve Open Dataset — Well 15/9-F-15 |
| Cross-project | offset-well-intelligence-crew Silver tables |
| Language | Python 3 |

---

## Setup

**Prerequisites:** Databricks workspace, Anthropic API key, Kaggle account

1. Data file included in `data/ROP_data.csv` — derived from Equinor Volve Open Dataset.
   Original Kaggle source: https://www.kaggle.com/datasets/ahmedelbashir99/drilling-log-dataset
   Upload to your Databricks Volume before running Phase 1.
2. Upload to Databricks Volume: `/Volumes/workspace/offset_well_crew/volve_data/`
3. Ensure `offset-well-intelligence-crew` Silver tables exist in your workspace
4. Run Phase 1 → Phase 2 → Phase 3 sequentially
5. Add your Anthropic API key in the configuration cell of Phase 2 and Phase 3

---

## Related Projects

| Repo | Pattern | Description |
|---|---|---|
| [offset-well-intelligence-crew](https://github.com/venkatchittoor/offset-well-intelligence-crew) | Prompt-based crew | Provides formation context Silver tables used by this advisor |
| [tool-calling-dq-agent](https://github.com/venkatchittoor/tool-calling-dq-agent) | True tool-calling | Claude writes its own SQL — the unconstrained version of this pattern |
| [drilling-npt-agent](https://github.com/venkatchittoor/drilling-npt-agent) | Prompt-based | NPT early warning — monitoring, not advisory |
| [data-incident-agent](https://github.com/venkatchittoor/data-incident-agent) | Prompt-based | Eyes/Brain/Hands monitoring pattern |

---

*Built by [VC](https://github.com/venkatchittoor) — Senior Consultant with O&G domain expertise, building at the intersection of data engineering and agentic AI.*
