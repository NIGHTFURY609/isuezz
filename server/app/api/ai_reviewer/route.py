from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import os
import httpx
import logging

router = APIRouter()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_URL = "https://api.cohere.ai/v1/chat"

MAX_CHARS_PER_FILE = 1000
MAX_FILES = 3

class FileMatch(BaseModel):
    file_name: str
    download_url: str
    match_score: float

class ReviewPayload(BaseModel):
    content_matches: list[FileMatch]
    filename_matches: list[FileMatch]
    owner: str
    repo: str
    issue_url: str
    issue_title: str
    issue_body: str

async def fetch_file_content(url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            res.raise_for_status()
            text = res.text[:MAX_CHARS_PER_FILE]
            return text + '\n... (truncated)' if len(res.text) > MAX_CHARS_PER_FILE else text
    except Exception as e:
        logging.error(f"Failed to fetch file content: {e}")
        return "Error fetching file."

@router.post("/ai-review")
async def review_issue(payload: ReviewPayload):
    if not COHERE_API_KEY:
        raise HTTPException(status_code=500, detail="Cohere API key is not configured")

    combined = payload.content_matches + payload.filename_matches
    top_files = sorted(combined, key=lambda x: x.match_score, reverse=True)[:MAX_FILES]

    file_infos = []
    for f in top_files:
        content = await fetch_file_content(f.download_url)
        file_infos.append({"file_name": f.file_name, "match_score": f.match_score, "content": content})

    prompt = f"""
Analyze this GitHub issue and relevant files:

Repository: {payload.owner}/{payload.repo}
Issue Title: {payload.issue_title}
Issue Description: {payload.issue_body[:500]}{'... (truncated)' if len(payload.issue_body) > 500 else ''}

Relevant Files:
"""
    for f in file_infos:
        prompt += f"""
File: {f['file_name']}
Match Score: {f['match_score']}
Key Content:
{f['content']}
"""

    prompt += """

Provide analysis in JSON format:
{
  "repository_analysis": {
    "purpose": "...",
    "tech_stack": ["..."],
    "issue_summary": "..."
  },
  "file_analysis": {
    "analyzed_files": [
      {"file_name": "...", "combined_probability": 0.0, "reason": "..."}
    ]
  },
  "recommendations": {
    "priority_order": ["..."],
    "specific_changes": "...",
    "additional_context": "..."
  }
}
"""

    try:
        headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
        body = {"message": prompt, "model": "command-r-plus"}
        async with httpx.AsyncClient() as client:
            response = await client.post(COHERE_API_URL, headers=headers, json=body)
            response.raise_for_status()
            content = response.json().get("text", "{}").strip()
            return {"reply": content}
    except Exception as e:
        logging.error(f"AI analysis failed: {e}")
        raise HTTPException(status_code=500, detail="AI analysis request failed")
