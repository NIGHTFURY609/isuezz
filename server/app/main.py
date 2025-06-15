from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.api.ai_reviewer.route import router as ai_reviewer_router
from app.api.ai_suggest.route import router as ai_suggest_router
from app.api.chatone_followup.route import router as chatone_followup_router

app = FastAPI(
    title="Issuezz",
    description="An AI-powered assistant for decoding open-source issues",
    version="1.0.0",
)

# CORS setup
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://issuezz.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers with appropriate prefixes
app.include_router(ai_reviewer_router, prefix="/api/ai_reviewer", tags=["AI Reviewer"])
app.include_router(ai_suggest_router, prefix="/api/ai_suggest", tags=["AI Suggest"])
app.include_router(chatone_followup_router, prefix="/api/chatone_followup", tags=["Chat Follow-up"])

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to Issuezz API!"}

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "Server is running!"}
