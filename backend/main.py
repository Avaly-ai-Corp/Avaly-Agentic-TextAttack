from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import re


app = FastAPI()


# ============================================
#  ADK API Configuration
#  - ADK_API_URL: URL of the ADK agent service
#  - ADK_APP_NAME: Name of the ADK application/agent
#  - ADK_USER_ID: User identifier for session creation
# ============================================
ADK_API_URL = "http://agent:8000"
ADK_APP_NAME = "multi_tool_agent"
ADK_USER_ID = "myusername"

# Allow CORS from React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def get_status():
    return {"message": "OK"}


@app.post("/api/create_session")
# No body is expected
async def create_session():
    # Use AsyncClient so we don't block the event loop
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{ADK_API_URL}/apps/{ADK_APP_NAME}/users/{ADK_USER_ID}/sessions",
            )
        except httpx.RequestError as exc:
            # If ADK server is down or unreachable, return a 502
            raise HTTPException(status_code=502, detail=f"ADK request failed: {exc}")

    # Forward ADK’s JSON response
    return resp.json()


@app.post("/api/run")
# Expected payload format:
# {
#     "appName": "multi_tool_agent",
#     "userId": "<userId>",
#     "sessionId": "<sessionId>",
#     "newMessage": {
#         "parts": [{ "text": "<inputText>" }],
#         "role": "user"
#     },
#     "streaming": false
# }
async def run(payload: dict):
    adk_url = f"{ADK_API_URL}/run"

    try:
        async with httpx.AsyncClient(timeout=10000.0) as client:
            resp = await client.post(adk_url, json=payload)
    except httpx.RequestError as exc:
        # network / timeout
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach ADK at {adk_url}: {exc}"
        )
    except Exception as e:
        # catch-all for unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while contacting ADK at {adk_url}: {e}"
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ADK returned {resp.status_code}: {resp.text}"
        )

    json_resp = resp.json()

    return json_resp