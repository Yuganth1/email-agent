import os
import re
import json
import httpx
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information about a person, company, or topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_document",
        "description": "Read and extract text content from the uploaded document.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why you are reading this document"}
            },
            "required": ["reason"]
        }
    },
    {
        "name": "draft_email",
        "description": "Draft a personalized, professional email based on gathered context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to_name": {"type": "string"},
                "to_email": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "tone": {"type": "string", "enum": ["formal", "warm", "direct"]}
            },
            "required": ["to_name", "subject", "body", "tone"]
        }
    }
]

async def tool_web_search(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": os.environ.get("TAVILY_API_KEY"),
                    "query": query,
                    "max_results": 5,
                    "search_depth": "basic"
                }
            )
            data = resp.json()
            results = []
            for r in data.get("results", []):
                results.append(f"- {r['title']}: {r['content'][:200]}")
            if results:
                return "\n".join(results)
    except Exception:
        pass
    return f"Search for '{query}': No results found."

def tool_read_document(doc_text: str, reason: str) -> str:
    if not doc_text:
        return "No document was uploaded."
    return f"Document content (reason: {reason}):\n\n{doc_text[:3000]}"

def tool_draft_email(to_name: str, to_email: str, subject: str, body: str, tone: str) -> dict:
    return {"to_name": to_name, "to_email": to_email or "—", "subject": subject, "body": body, "tone": tone}

async def run_agent(target: str, goal: str, doc_text: str):
    system_prompt = """You are an intelligent outreach agent. Your job is to:
1. Research the target person/company using web_search (always do at least one search)
2. Read any uploaded document if one was provided using read_document
3. Draft a highly personalized email using draft_email

Always follow this order: search first, read document if uploaded, then draft email last.
Think step by step and briefly explain your reasoning before each tool call.
Make the email reference specific things you discovered."""

    user_message = f"""Target: {target}
Goal: {goal}
Document uploaded: {"Yes — use read_document to access it" if doc_text else "No"}

Research this target and draft a personalized outreach email."""

    messages = [{"role": "user", "content": user_message}]
    final_email = None

    yield f"data: {json.dumps({'type': 'status', 'text': 'Agent starting...'})}\n\n"
    await asyncio.sleep(0.1)

    for turn in range(10):
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                clean = block.text
                clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
                clean = re.sub(r'#{1,3}\s?', '', clean)
                clean = re.sub(r'\*(.*?)\*', r'\1', clean)
                yield f"data: {json.dumps({'type': 'thinking', 'text': clean})}\n\n"
                await asyncio.sleep(0.05)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'input': tool_input})}\n\n"
                await asyncio.sleep(0.2)

                if tool_name == "web_search":
                    result = await tool_web_search(tool_input["query"])
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': result[:400] + '...' if len(result) > 400 else result})}\n\n"

                elif tool_name == "read_document":
                    result = tool_read_document(doc_text, tool_input["reason"])
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': f'Document read — {len(doc_text)} chars extracted'})}\n\n"

                elif tool_name == "draft_email":
                    result = tool_draft_email(**tool_input)
                    final_email = result
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': 'Email drafted'})}\n\n"
                    result = json.dumps(result)
                else:
                    result = "Tool not found"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result if isinstance(result, str) else json.dumps(result)
                })
                await asyncio.sleep(0.1)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    if final_email:
        yield f"data: {json.dumps({'type': 'email', 'data': final_email})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/run-agent")
async def run_agent_endpoint(
    target: str = Form(...),
    goal: str = Form(...),
    document: Optional[UploadFile] = File(None)
):
    doc_text = ""
    if document and document.filename:
        content = await document.read()
        try:
            doc_text = content.decode("utf-8")
        except Exception:
            doc_text = f"[Binary file: {document.filename}]"

    return StreamingResponse(
        run_agent(target, goal, doc_text),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_index():
    return FileResponse(str(frontend_dir / "index.html"))

if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir)), name="static")
