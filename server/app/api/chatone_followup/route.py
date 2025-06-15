from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import re
import cohere
from dotenv import load_dotenv
import logging

load_dotenv()

router = APIRouter()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not set in environment")

co = cohere.Client(COHERE_API_KEY)

# Pydantic models for request
class ChatMessage(BaseModel):
    type: str  # 'bot' or 'user'
    content: str

class UserProfile(BaseModel):
    name: str
    bio: str
    public_repos: int
    hireable: bool

class Repository(BaseModel):
    name: str
    language: str
    topics: list[str]

class TechnicalContext(BaseModel):
    languages: list[str]
    topics: list[str]

class ChatContext(BaseModel):
    userProfile: UserProfile
    previousMessages: list[ChatMessage]
    currentQuery: str
    userRepos: list[Repository]
    technicalContext: TechnicalContext

# Helper to categorize guidance needs
def analyze_user_query(query: str) -> list[str]:
    patterns = {
        'setup': re.compile(r"(how to|help|can you|where do I) (start|begin|setup|set up|initialize)", re.I),
        'files': re.compile(r"(which|what|where) (files?|code|changes|modify)", re.I),
        'workflow': re.compile(r"(steps|process|workflow|how do I|what should I)", re.I),
        'testing': re.compile(r"(test|verify|check|validate)", re.I),
        'submission': re.compile(r"(submit|PR|pull request|contribute)", re.I),
        'explanation': re.compile(r"(explain|understand|what does|mean|confused|unclear)", re.I),
        'error': re.compile(r"(error|problem|issue|not working|failed)", re.I),
        'conceptual': re.compile(r"(concept|theory|principle|how does|why does)", re.I)
    }
    categories = [cat for cat, pat in patterns.items() if pat.search(query)]
    return categories or ['general']

# Helper to extract issue context
def extract_issue_context(messages: list[ChatMessage]) -> str:
    for msg in messages:
        if msg.type == 'bot' and 'recommended issues' in msg.content:
            match = re.search(r'Issue Title: (.*?)\nURL: (.*?)(\n|$)', msg.content)
            if match:
                return f"Working on issue: {match.group(1)}\nURL: {match.group(2)}"
            return msg.content
    return ''

@router.post("/api/chatone_followup")
async def chatone_followup(request: Request):
    try:
        data = await request.json()
        context = ChatContext(**data)

        guidance_needed = analyze_user_query(context.currentQuery)
        issue_context = extract_issue_context(context.previousMessages)
        chat_history = "\n\n".join(f"{m.type.upper()}: {m.content}" for m in context.previousMessages)

        prompt = f"""
As an experienced open source mentor helping a beginner developer, provide detailed guidance based on their question.

DEVELOPER CONTEXT:
- Experience Level: Beginner
- Known Languages: {', '.join(context.technicalContext.languages)}
- Interests/Topics: {', '.join(context.technicalContext.topics)}
- Public Repos: {context.userProfile.public_repos}

GUIDANCE CATEGORIES NEEDED: {', '.join(guidance_needed)}

ISSUE CONTEXT:
{issue_context}

CHAT HISTORY:
{chat_history}

CURRENT QUESTION:
{context.currentQuery}

Based on their question, provide:
1. Clear, direct answer to their specific question
2. Step-by-step instructions appropriate for beginners
3. Explanation of any technical terms or concepts
4. Specific file locations and code areas to work with
5. Common pitfalls and how to avoid them
6. Testing and verification steps
7. Relevant documentation or learning resources

RESPONSE GUIDELINES:
1. Use simple, clear language suitable for beginners
2. Break down complex tasks into small, manageable steps
3. Provide examples using their known programming languages
4. Include error handling and debugging guidance
5. Suggest ways to verify work
6. Be encouraging and supportive
7. Use a friendly tone with occasional emojis
8. Address potential confusion points proactively

Remember: Keep explanations beginner-friendly and maintain an encouraging tone throughout.
"""
        # Send to Cohere
        response = co.generate(
            model="command-xlarge-beta",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        reply_text = response.generations[0].text.strip()
        return JSONResponse(status_code=200, content={"reply": reply_text})

    except Exception as e:
        logging.exception("Chat follow-up failed")
        raise HTTPException(status_code=500, detail=str(e))
