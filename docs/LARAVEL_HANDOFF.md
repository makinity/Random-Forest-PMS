# FastAPI ML Service — Laravel Integration Handoff

**Status: ✅ FastAPI is built, tested, and working.**
This document is for the Laravel developer to integrate Smart PMS with the ML microservice.

---

## What Is Running

A FastAPI microservice that:
- Connects to the **same MySQL database** as Laravel (read-only except `ml_kpi_predictions`)
- Trains a Random Forest model on `employee_performance_snapshots` (315 seeded rows)
- Writes ranked employee predictions into `ml_kpi_predictions`
- Serves those predictions to Laravel on demand

---

## Verified Endpoints

### `POST /train`
Trains the model and populates `ml_kpi_predictions`. Run this once after seeding, and again whenever new IPCR data is released.

**Request:** no body

**Response:**
```json
{
  "status": "ok",
  "rows_trained": 315,
  "indicators_upserted": 7,
  "model_version": "1.0.0"
}
```

---

### `POST /suggest-employees`
Returns ranked employee recommendations for a given success indicator.

**Request body:**
```json
{
  "uwp_success_indicator_id": 1,
  "performance_period_id": 1
}
```

**Response:**
```json
{
  "feasibility_label": "achievable",
  "feasibility_probability": 0.9993,
  "risk_level": "Low",
  "recommendations": [
    {
      "employee_id": 9,
      "fit_score": 99.9,
      "fit_label": "Strong fit",
      "risk_level": "Low",
      "warning": false
    },
    {
      "employee_id": 3,
      "fit_score": 97.0,
      "fit_label": "Strong fit",
      "risk_level": "Low",
      "warning": false
    }
  ]
}
```

**Field contracts (AssignModal reads these exact names):**

| Field | Type | Values |
|---|---|---|
| `feasibility_label` | string | `achievable` / `at_risk` / `unrealistic` |
| `feasibility_probability` | float | `0.0` – `1.0` |
| `risk_level` | string | `Low` / `Medium` / `High` |
| `fit_score` | float | `0` – `100` |
| `fit_label` | string | `Strong fit` / `Moderate fit` / `Weak fit` |
| `warning` | boolean | `true` if employee prediction is `unrealistic` |

---

### `GET /health`
**Response:**
```json
{
  "db_connected": true,
  "model_loaded": true,
  "snapshot_rows": 315
}
```

---

## How Laravel Should Call FastAPI

FastAPI is **not** exposed to the browser. Laravel proxies all calls internally.

### 1. Add to Laravel `.env`
```
FASTAPI_URL=http://127.0.0.1:8000
```
> During development with ngrok, replace the value with your ngrok URL e.g. `https://abc123.ngrok-free.app`

### 2. Laravel Controller
```php
use Illuminate\Support\Facades\Http;

public function suggestions(Request $request)
{
    $response = Http::post(env('FASTAPI_URL') . '/suggest-employees', [
        'uwp_success_indicator_id' => (int) $request->indicator_id,
        'performance_period_id'    => (int) $request->period_id,
    ]);

    if ($response->failed()) {
        return response()->json(['error' => 'ML service unavailable'], 503);
    }

    return $response->json();
}
```

### 3. Laravel Route
```php
Route::get('/supervisor/uwp/suggestions', [UwpController::class, 'suggestions']);
```

### 4. AssignModal (React) — replace `suggestIndicators()` simulation
```js
const res = await axios.get('/supervisor/uwp/suggestions', {
    params: { indicator_id: indicatorId, period_id: periodId }
});
// res.data.recommendations → array of { employee_id, fit_score, fit_label, risk_level, warning }
// res.data.feasibility_label, res.data.feasibility_probability, res.data.risk_level → indicator-level
```

---

## Development Bridge — ngrok

Since FastAPI runs locally on a separate machine/port during development, use **ngrok** to expose it:

```bash
# Install ngrok, then:
ngrok http 8000
```

ngrok gives you a public URL like `https://abc123.ngrok-free.app`.
Set that as `FASTAPI_URL` in Laravel's `.env` — no firewall changes needed.

> **Important:** ngrok URL changes every restart (free tier). For stable dev, pin it with a static domain or use a paid ngrok plan.

---

## Database Tables (FastAPI writes, Laravel reads)

### `ml_kpi_predictions` — written by FastAPI, read by Laravel
| Column | Written by FastAPI |
|---|---|
| `uwp_success_indicator_id` | ✅ |
| `performance_period_id` | ✅ |
| `feasibility_label` | ✅ |
| `feasibility_probability` | ✅ |
| `risk_level` | ✅ |
| `recommendations` | ✅ JSON array (ranked by fit_score desc, one entry per employee) |
| `model_version` | ✅ `"1.0.0"` |
| `generated_at` | ✅ timestamp |

### `employee_performance_snapshots` — read-only by FastAPI
Populated by `IpcrObserver` on PMT IPCR release, and by `MlTrainingDataSeeder` (315 rows seeded).

---

## When to Call `/train`

| Trigger | Action |
|---|---|
| After `php artisan db:seed --class=MlTrainingDataSeeder` | Call `POST /train` once |
| After PMT releases an IPCR (`IpcrObserver` fires) | Optionally queue a `POST /train` job |
| Manual retrain needed | Call `POST /train` directly |

Laravel can trigger retraining via a queued job:
```php
Http::post(env('FASTAPI_URL') . '/train');
```

---

## FastAPI Project Location
```
C:\Python\ML\RandomForest\
├── app/
│   ├── main.py                  ← entry point
│   ├── db.py                    ← MySQL connection (SQLAlchemy)
│   ├── config/settings.py       ← reads .env
│   ├── store/model_registry.py  ← joblib model load/save
│   ├── schemas/uwp.py           ← request/response types
│   ├── train/train_model.py     ← RF training + UPSERT logic
│   ├── infer/predict.py         ← /suggest-employees logic
│   └── api/routes.py            ← endpoint wiring
├── .env                         ← DB credentials (fill in for your DB)
└── requirements.txt
```

Start the service:
```bash
python -m uvicorn app.main:app --reload
```
