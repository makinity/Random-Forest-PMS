# ML Integration — Smart PMS × FastAPI
**Random Forest KPI Feasibility Model**
*Attach this file to the FastAPI project for full context.*

---

## Research Objectives

| # | Objective | Status |
|---|---|---|
| 6.1 | Analyze historical performance data and task attributes | ✅ Table designed + seeded |
| 6.2 | Validate and predict the feasibility of proposed KPI targets | ✅ RF model trained (Colab) |
| 6.3 | Classify KPIs as Achievable, At Risk, or Unrealistic | ✅ `feasibility_label` output |
| 6.4 | Provide probability percentages and risk levels per KPI | ✅ `feasibility_probability` + `risk_level` |
| 6.5 | Recommend appropriate employee for a specific KPI | ✅ `fit_score` + `fit_label` per employee |

**Panel term → System term mapping:**

| Panel/Docs Term | Smart PMS Term |
|---|---|
| KPI | Success Indicator (`uwp_success_indicators.indicator_text`) |
| KPI Target | `target_quantity` + `target_timeline` on the indicator |
| KPI Feasibility | Whether the target is realistic for a given employee (RF output) |
| Achievable | `adjectival_rating` Outstanding / Very Satisfactory (score ≥ 4.0) |
| At Risk | `adjectival_rating` Satisfactory (score 3.0–3.99) |
| Unrealistic | `adjectival_rating` Unsatisfactory / Poor (score < 3.0) |
| Recommend Employee | Rank employees by `fit_score` for assignment in UWP Editor |
| Historical Performance | `employee_performance_snapshots` table |

---

## System Architecture

```
Smart PMS (Laravel + React)
│
├── Supervisor UWP Editor
│     └── AssignModal
│           ├── [Current] hardcoded suggestIndicators() simulation
│           └── [Target]  GET /pmt/uwp/suggestions?indicator_id=X
│                              ↓
│                         Laravel controller
│                              ↓ HTTP POST
│                         FastAPI /suggest-employees
│                              ↓
│                         reads ml_kpi_predictions (MySQL)
│                              ↑
│                         FastAPI writes predictions after training
│
└── MySQL Database (shared)
      ├── employee_performance_snapshots  ← FastAPI reads for training
      └── ml_kpi_predictions              ← FastAPI writes predictions
```

**FastAPI has read/write access to two tables only:**
- `employee_performance_snapshots` — read-only (training data)
- `ml_kpi_predictions` — write (store prediction output)

---

## Database Tables

### Table 1 — `employee_performance_snapshots`
**Purpose:** Denormalized ML training data. One row = one employee × one indicator × one performance cycle.
Auto-populated by `IpcrObserver` when PMT releases an IPCR. Also seeded synthetically.

| Column | Type | Source | ML Role |
|---|---|---|---|
| `employee_id` | FK → users | `ipcrs.employee_id` | Identifier |
| `performance_period_id` | FK → performance_periods (nullable) | `ipcrs.performance_period_id` | Identifier |
| `ipcr_id` | FK → ipcrs (nullable) | `ipcrs.id` | Identifier |
| `uwp_success_indicator_id` | FK → uwp_success_indicators (nullable) | `ipcr_items.indicator_id` | Feature |
| `position` | string | `users.position` | Feature — seniority |
| `office_name` | string | `offices.name` | Feature — context |
| `indicator_text` | text | `ipcr_items.indicator_text` | Feature — NLP similarity |
| `function_type` | string | `uwp_functions.function_type` | Feature — `core`/`support` |
| `mfo_title` | string | `uwp_mfos.title` | Feature — work category |
| `target_quantity` | integer | `uwp_success_indicators.target_quantity` | Feature — demand |
| `target_timeline_days` | integer | parsed from `target_timeline` | Feature — urgency |
| `office_size` | integer | COUNT users in office | Feature — workload context |
| `employee_count_assigned` | integer | COUNT assignments on indicator | Feature — shared load |
| `current_workload_count` | integer | COUNT indicators assigned this period | Feature — employee load |
| `previous_final_score` | decimal | last period's `final_score` | Feature — trend |
| `previous_adjectival_rating` | string | last period's rating | Feature — trend |
| `was_flagged_for_calibration` | boolean | `accomplishment_submissions.dept_head_flagged_for_calibration` | Feature — quality signal |
| `final_score` | decimal | `ipcrs.pmt_adjusted_score` ?? `final_score` | **Label input** |
| `adjectival_rating` | string | PMT-adjusted rating | Label input |
| `feasibility_label` | string | derived from `final_score` | **ML Target Label** |

**Feasibility label derivation:**
```
final_score >= 4.0  →  achievable
final_score >= 3.0  →  at_risk
final_score <  3.0  →  unrealistic
```

**When is it populated?**
- **Real data:** `IpcrObserver::updated()` fires when `ipcrs.status = 'released_by_pmt'`
- **Synthetic seed:** `php artisan db:seed --class=MlTrainingDataSeeder` (315 rows, 5 passes × 9 employees × 7 indicators)

---

### Table 2 — `ml_kpi_predictions`
**Purpose:** Stores FastAPI prediction output. Laravel reads this table to serve the AssignModal — no live API call needed per request.

| Column | Type | Description |
|---|---|---|
| `uwp_success_indicator_id` | FK → uwp_success_indicators | Which KPI was assessed |
| `performance_period_id` | FK → performance_periods | Which period |
| `feasibility_label` | string | `achievable` / `at_risk` / `unrealistic` (Obj. 6.3) |
| `feasibility_probability` | decimal(5,4) | e.g. `0.8700` = 87% confidence (Obj. 6.4) |
| `risk_level` | string | `Low` / `Medium` / `High` (Obj. 6.4) |
| `recommendations` | JSON | Ranked employee list (Obj. 6.5) — see structure below |
| `model_version` | string | e.g. `1.0.0` |
| `generated_at` | timestamp | When FastAPI ran the prediction |

**`recommendations` JSON structure** (maps directly to AssignModal fields):
```json
[
  {
    "employee_id": 3,
    "fit_score": 93.2,
    "fit_label": "Strong fit",
    "feasibility_label": "achievable",
    "feasibility_probability": 0.9319,
    "risk_level": "Low",
    "warning": false
  },
  {
    "employee_id": 7,
    "fit_score": 52.1,
    "fit_label": "Moderate fit",
    "feasibility_label": "at_risk",
    "feasibility_probability": 0.5210,
    "risk_level": "Medium",
    "warning": false
  }
]
```

**Unique constraint:** one prediction per `(uwp_success_indicator_id, performance_period_id)` — FastAPI overwrites on retrain.

---

## Data Flow

```
1. REAL DATA PATH
   Supervisor creates UWP → assigns indicators to employees
        ↓
   Employee submits IPCR → Dept Head reviews → PMT scores & releases
        ↓
   IpcrObserver fires → writes to employee_performance_snapshots
        ↓
   FastAPI reads snapshots → trains Random Forest → writes ml_kpi_predictions

2. SYNTHETIC DATA PATH (for development/testing)
   php artisan db:seed --class=MlTrainingDataSeeder
        ↓
   315 rows in employee_performance_snapshots
        ↓
   FastAPI reads → trains → writes ml_kpi_predictions

3. PREDICTION SERVING PATH
   Supervisor opens AssignModal for a success indicator
        ↓
   Laravel: SELECT * FROM ml_kpi_predictions WHERE uwp_success_indicator_id = ?
        ↓
   Returns recommendations JSON → AssignModal renders fit scores, risk, warning
```

---

## What Historical Data Is Compared

The Random Forest compares an employee's **past performance on similar work** against the **current indicator's demands**:

| Compared | Historical Source | Current Source |
|---|---|---|
| Work type | `function_type` from past IPCR items | `uwp_success_indicators` function_type |
| Workload demand | `target_quantity` past vs current | Current indicator's `target_quantity` |
| Timeline pressure | `target_timeline_days` past vs current | Current indicator's parsed timeline |
| Employee load at time | `current_workload_count` (how many indicators assigned) | Current period assignment count |
| Office context | `office_size` when assigned | Current office size |
| Outcome | `final_score` + `adjectival_rating` (what actually happened) | Prediction target |
| Trend | `previous_final_score` vs `final_score` | Improvement/decline signal |
| Calibration history | `was_flagged_for_calibration` | Risk signal |

---

## Google Colab Output → FastAPI Input

The CSV `ml_kpi_predictions_generated.csv` from Colab has this structure:

```
employee_id | uwp_success_indicator_id | performance_period_id |
feasibility_label | feasibility_probability | risk_level |
fit_score | fit_label | warning
```

**FastAPI import script** reads this CSV and populates `ml_kpi_predictions`:
```python
# Group by indicator → build recommendations JSON per indicator
grouped = df.groupby("uwp_success_indicator_id").apply(lambda g:
    g.sort_values("fit_score", ascending=False)[
        ["employee_id","fit_score","fit_label",
         "feasibility_label","feasibility_probability","risk_level","warning"]
    ].to_dict("records")
)

for indicator_id, recs in grouped.items():
    top = recs[0]  # highest fit score determines the indicator-level feasibility
    db.execute("""
        INSERT INTO ml_kpi_predictions
        (uwp_success_indicator_id, performance_period_id,
         feasibility_label, feasibility_probability, risk_level,
         recommendations, model_version, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
          feasibility_label = VALUES(feasibility_label),
          feasibility_probability = VALUES(feasibility_probability),
          risk_level = VALUES(risk_level),
          recommendations = VALUES(recommendations),
          generated_at = NOW()
    """, (indicator_id, period_id,
          top["feasibility_label"], top["feasibility_probability"],
          top["risk_level"], json.dumps(recs), "1.0.0"))
```

---

## FastAPI Endpoints Needed

| Endpoint | Method | Description |
|---|---|---|
| `/train` | POST | Re-train RF on latest `employee_performance_snapshots`, write to `ml_kpi_predictions` |
| `/suggest-employees` | POST | Return ranked employee recommendations for a given indicator (reads `ml_kpi_predictions`) |
| `/predict` | POST | On-demand prediction for a single employee × indicator pair |
| `/health` | GET | Check DB connection + model status |

### `/suggest-employees` input (sent by Laravel):
```json
{
  "uwp_success_indicator_id": 4,
  "performance_period_id": 1
}
```

### `/suggest-employees` output (returned to AssignModal):
```json
{
  "feasibility_label": "achievable",
  "feasibility_probability": 0.93,
  "risk_level": "Low",
  "recommendations": [
    {
      "employee_id": 3,
      "fit_score": 93.2,
      "fit_label": "Strong fit",
      "risk_level": "Low",
      "warning": false
    }
  ]
}
```

---

## Laravel Integration (AssignModal)

**Current state:** `suggestIndicators()` in `AssignModal.jsx` is a hardcoded seed-based simulation.

**Target state:** Replace with real API call via Laravel proxy:

```php
// Route: GET /supervisor/uwp/suggestions?indicator_id=X
public function suggestions(Request $request)
{
    $prediction = DB::table('ml_kpi_predictions')
        ->where('uwp_success_indicator_id', $request->indicator_id)
        ->where('performance_period_id', $request->period_id)
        ->first();

    return response()->json($prediction
        ? json_decode($prediction->recommendations)
        : []
    );
}
```

**AssignModal field mapping:**

| AssignModal field | `ml_kpi_predictions.recommendations` field |
|---|---|
| `ai.successProb` | `fit_score` |
| `ai.risk` | `risk_level` |
| `ai.warning` | `warning` |
| `fitScore` | `fit_score` |
| `fitLabel` | `fit_label` |
| `fitColor` | derived: Low=green, Medium=yellow, High=red |

---

## Files Reference

| File | Purpose |
|---|---|
| `database/migrations/2026_06_10_000001_create_employee_performance_snapshots_table.php` | Original snapshot table |
| `database/migrations/2026_06_11_000001_revamp_employee_performance_snapshots_for_ml.php` | Added ML columns |
| `database/migrations/2026_06_11_000002_create_ml_kpi_predictions_table.php` | Prediction output table |
| `database/migrations/2026_06_11_000003_make_performance_period_nullable_in_snapshots.php` | Allow synthetic rows |
| `database/seeders/MlTrainingDataSeeder.php` | 315 synthetic training rows |
| `app/Observers/IpcrObserver.php` | Auto-populates snapshots on IPCR release |
| `app/Models/EmployeePerformanceSnapshot.php` | Eloquent model |
| `docs/google-colab.md` | Colab notebook code (train + export) |

---

## Current Status

| Component | Status |
|---|---|
| `employee_performance_snapshots` table | ✅ Migrated + 315 seeded rows |
| `ml_kpi_predictions` table | ✅ Migrated, empty (FastAPI writes here) |
| `IpcrObserver` auto-population | ✅ Fires on PMT IPCR release |
| Google Colab RF training | ✅ Trains + exports `ml_kpi_predictions_generated.csv` |
| FastAPI `/train` + DB write | 🔲 To build |
| FastAPI `/suggest-employees` | 🔲 To build |
| Laravel proxy endpoint | 🔲 To build |
| AssignModal real API integration | 🔲 Replace `suggestIndicators()` simulation |

---

*Last updated: 2026-06-11*
