from fastapi import FastAPI, UploadFile, File, HTTPException
import base64
import httpx

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.on_event("startup")
async def startup():
    # reuse one client (connection pooling)
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))

@app.on_event("shutdown")
async def shutdown():
    await app.state.client.aclose()

@app.post("/caption/")
async def caption_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": "llava",
        "prompt": "Describe this image in one sentence.",
        "images": [image_base64],
        "stream": False
    }

    try:
        resp = await app.state.client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama request timed out.")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {e}")

    caption = (data.get("response") or "").strip()
    if not caption:
        raise HTTPException(status_code=500, detail="Empty caption returned by model.")
    return {"caption": caption}
