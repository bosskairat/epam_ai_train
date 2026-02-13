Orders Management API (FastAPI)

Quick start

- Create virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run app:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Seed happens automatically on first import.

API endpoints

- POST /orders  — create order
- GET /orders?page=1&limit=10&status=created&min_amount=10&max_amount=100&start_date=2024-01-01&end_date=2024-12-31

API documentation

Base URL

- http://127.0.0.1:8000

Interactive docs

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

POST /orders

Create a new order.

Request body

```json
{
  "status": "created",
  "amount": 123.45
}
```

Response 200

```json
{
  "id": 1,
  "status": "created",
  "amount": 123.45,
  "created_at": "2026-02-13T06:42:20.256836+00:00"
}
```

Error responses

- 400: Amount must be non-negative
- 400: Invalid status

GET /orders

List orders with pagination and optional filters.

Query parameters

- page: int, default 1, min 1
- limit: int, default 10, min 1, max 100
- status: one of created | processing | completed | cancelled
- min_amount: float, min 0
- max_amount: float, min 0
- start_date: ISO datetime or date string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- end_date: ISO datetime or date string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)

Example request

```bash
curl "http://127.0.0.1:8000/orders?page=1&limit=3&status=completed&min_amount=50&start_date=2024-01-01"
```

Response 200

```json
{
  "items": [
    {
      "id": 17,
      "status": "completed",
      "amount": 356.07,
      "created_at": "2025-07-02T11:39:00.851922+00:00"
    },
    {
      "id": 23,
      "status": "completed",
      "amount": 287.62,
      "created_at": "2025-11-12T11:39:00.851922+00:00"
    }
  ],
  "total": 82,
  "page": 1,
  "limit": 3,
  "pages": 28,
  "has_next": true,
  "has_prev": false
}
```

Error responses

- 400: Invalid date format. Use ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
- 400: start_date must be less than or equal to end_date
- 422: Validation error for invalid query params (e.g., page=0 or limit=101)

Tests

```bash
pytest -q
```
