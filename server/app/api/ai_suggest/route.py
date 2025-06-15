from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import cohere
from dotenv import load_dotenv
import json
import logging

load_dotenv()

router = APIRouter()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not set in environment")

co = cohere.Client(COHERE_API_KEY)

@router.post("/api/ai_suggest")
async def suggest_issues(request: Request):
    try:
        body = await request.json()

        user_languages = list({repo.get("language") for repo in body["repositories"] if repo.get("language")})
        user_topics = list({topic for repo in body["repositories"] for topic in repo.get("topics", [])})
        available_issues = [issue for issue in body["repoissues"] if not issue.get("pull_request")]

        if not available_issues:
            return JSONResponse(status_code=200, content={"reply": {"recommendations": []}})

        issue_descriptions = "\n".join(
            [
                f"Issue #{i+1}:\n"
                f"- Title: {issue['title']}\n"
                f"- Repository: {body['issue_owner']}/{body['issue_repo']}\n"
                f"- Description: {issue['body']}\n"
                f"- Labels: {', '.join(issue.get('labels', []))}\n"
                f"- URL: https://github.com/{body['issue_owner']}/{body['issue_repo']}/issues/{issue['number']}"
                for i, issue in enumerate(available_issues)
            ]
        )

        prompt = f"""
OPEN-SOURCE CONTRIBUTION MATCHER

## DEVELOPER PROFILE
- Programming Languages: {', '.join(user_languages)}
- Technical Interests: {', '.join(user_topics)}
- Experience Level: Beginner

## AVAILABLE ISSUES (Total: {len(available_issues)})
{issue_descriptions}

## RECOMMENDATION OBJECTIVE
Analyze the available issues and recommend the most suitable ones for a beginner developer.
For each recommended issue, provide:
1. Clear difficulty assessment
2. Brief summary of required changes
3. Key files that need modification
4. Essential skills needed
5. Estimated time commitment

Important constraints:
- Recommend MAX 3 issues that best match the developer's skills
- Focus on beginner-friendly issues
- Keep initial descriptions concise but informative
- If no suitable issues exist, return empty recommendations

FORMAT YOUR RESPONSE AS JSON following the structure below:
{{
  "recommendations": [
    {{
      "issue_title": "Exact Issue Title",
      "issue_url": "Full GitHub Issue URL",
      "difficulty_level": "Beginner/Intermediate/Advanced",
      "quick_summary": "One-sentence overview of what needs to be done",
      "key_skills_needed": ["2-3 main skills required"],
      "main_files": ["2-3 key files to modify"],
      "estimated_time": "Rough time estimate for beginners",
      "why_recommended": "Brief explanation of why this matches their skills"
    }}
  ]
}}
"""

        response = co.generate(
            model="command-r",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )

        raw_text = response.generations[0].text.strip()
        cleaned_json = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned_json)

        recommendations = parsed.get("recommendations", [])[:3]

        return JSONResponse(status_code=200, content={"reply": {"recommendations": recommendations}})

    except Exception as e:
        logging.exception("AI Suggestion failed")
        raise HTTPException(status_code=500, detail=str(e))
