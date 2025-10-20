from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import time

app = FastAPI(title="Guardian Integration API", version="0.1.0")

EVENTS: List[Dict[str, Any]] = []

@app.post("/api/events")
async def receive_event(req: Request):
    data = await req.json()
    data["received_at"] = int(time.time()*1000)
    EVENTS.append(data)
    # Simula lógica do Guardian: ao reconhecer uma pessoa, gera um 'advisory' base
    advisory = {
        "message": f"Olá, {data.get('person','usuário')}! Antes de concluir um Pix para apostas, considere investir esse valor.",
        "suggested_return_apr_percent": 8.0,
        "cta": "Abrir simulação XP"
    }
    return JSONResponse({"ok": True, "stored": data, "advisory": advisory})

@app.get("/api/events")
async def list_events():
    return {"count": len(EVENTS), "events": EVENTS[-50:]}

@app.get("/health")
async def health():
    return {"status": "ok", "events": len(EVENTS)}
