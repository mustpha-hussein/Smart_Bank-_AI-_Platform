# SmartBank AI

An end-to-end machine-learning web application for loan eligibility and fraud-risk predictions.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8010 --reload
```

Open http://127.0.0.1:8010. The FastAPI documentation is at http://127.0.0.1:8010/docs.

## API

- `GET /api/health` — model readiness check
- `POST /api/loan/predict` — loan prediction
- `POST /api/fraud/predict` — fraud prediction

The UI is served by the same FastAPI process, so it sends requests directly to the two model endpoints. The fraud artifact was originally serialized from a notebook; the service restores its custom preprocessing classes before loading it.
