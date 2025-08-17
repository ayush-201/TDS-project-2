
# Data Analyst Agent – Render Deployment

## Run locally (Docker)
```bash
docker build -t data-analyst-agent .
docker run -p 8000:8000 -e GEMINI_API_KEY=YOUR_KEY data-analyst-agent
```

## Endpoint
```
POST /api/
Form fields:
 - questions.txt (required): the prompt/questions
 - (optional) additional files: data.csv, image.png, etc.
```

### Example
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions.txt=@question.txt" \
  -F "data.csv=@data.csv"
```

## Health check
```
GET /healthz
```

## Render.com
- Create a **Web Service** from your GitHub repo.
- Runtime: **Docker** (uses Dockerfile).
- Environment Variable: `GEMINI_API_KEY`.
- It will listen on `$PORT` automatically.
```
