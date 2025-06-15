from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import cohere
from dotenv import load_dotenv
import logging

load_dotenv()

router = APIRouter()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not set in environment")

co = cohere.Client(COHERE_API_KEY)

# Pydantic models
class ChatMessage(BaseModel):
    type: str  # 'bot' or 'user'
    content: str

class FileContent(BaseModel):
    name: str
    content: str

class TechnicalContext(BaseModel):
    languages: list[str]
    topics: list[str]

class RepositoryAnalysis(BaseModel):
    tech_stack: list[str]
    purpose: str

class RecommendationsContext(BaseModel):
    specific_changes: str

class AnalysisContext(BaseModel):
    repository_analysis: RepositoryAnalysis
    recommendations: RecommendationsContext

class ChatTwoPayload(BaseModel):
    previousMessages: list[ChatMessage]
    currentQuery: str
    fileContents: list[FileContent]
    analysisContext: AnalysisContext
    requestType: str  # 'code_explanation', 'workflow', or other
    technicalContext: TechnicalContext

# Helper functions
def generate_code_explanation_prompt(file_contents, query, context):
    files_text = "\n".join(
        f"File: {f.name}\nContent:\n{f.content}" for f in file_contents
    )
    return (
        f"As a developer experienced in {', '.join(context.repository_analysis.tech_stack)}, "
        f"explain the following code in the context of {context.repository_analysis.purpose}.\n\n"
        f"User Question: {query}\n\n"
        f"Relevant Files:\n{files_text}\n\n"
        "Provide a detailed explanation that:\n"
        "1. Addresses the specific question about the code\n"
        "2. Explains the purpose and functionality of relevant code sections\n"
        "3. Highlights any important patterns or practices used\n"
        "4. Connects the code to the overall project context\n"
        "5. Suggests any potential improvements or considerations"
    )


def generate_workflow_prompt(query, context, previous_messages):
    prev_text = "\n".join(f"{m.type.upper()}: {m.content}" for m in previous_messages)
    return (
        f"As a technical advisor familiar with {', '.join(context.repository_analysis.tech_stack)}, "
        "help understand the workflow of this project.\n\n"
        f"Project Context:\n{context.repository_analysis.purpose}\n\n"
        f"Current Question: {query}\n\n"
        "Previous Discussion Context:\n"
        f"{prev_text}\n\n"
        f"Provide guidance that:\n"
        f"1. Explains the relevant workflow aspects\n"
        f"2. Connects to the project's overall architecture\n"
        f"3. References specific recommendations: {context.recommendations.specific_changes}\n"
        f"4. Suggests next steps or areas to focus on\n"
        f"5. Highlights best practices and potential improvements"
    )


def generate_general_prompt(query, context, previous_messages):
    prev_text = "\n".join(f"{m.type.upper()}: {m.content}" for m in previous_messages)
    return (
        f"As a technical advisor for this {', '.join(context.repository_analysis.tech_stack)} project, "
        "address the following question.\n\n"
        f"Project Context:\n{context.repository_analysis.purpose}\n\n"
        f"Current Question: {query}\n\n"
        "Previous Discussion Context:\n"
        f"{prev_text}\n\n"
        "Provide a response that:\n"
        "1. Directly addresses the question\n"
        "2. Connects to the project context\n"
        "3. References relevant technical aspects\n"
        "4. Suggests practical next steps\n"
        "5. Maintains consistency with previous answers"
    )

@router.post("/api/chattwo_followup")
async def chattwo_followup(request: Request):
    try:
        data = await request.json()
        payload = ChatTwoPayload(**data)

        # Select prompt based on requestType
        if payload.requestType == 'code_explanation':
            prompt = generate_code_explanation_prompt(
                payload.fileContents,
                payload.currentQuery,
                payload.analysisContext
            )
        elif payload.requestType == 'workflow':
            prompt = generate_workflow_prompt(
                payload.currentQuery,
                payload.analysisContext,
                payload.previousMessages
            )
        else:
            prompt = generate_general_prompt(
                payload.currentQuery,
                payload.analysisContext,
                payload.previousMessages
            )

        # System message
        system_content = (
            f"You are an expert {', '.join(payload.technicalContext.languages)} developer and technical advisor. " +
            ("Provide detailed, educational code explanations with examples and best practices."
             if payload.requestType == 'code_explanation'
             else "Offer clear, actionable guidance while maintaining context from previous messages.")
        )

        # Call Cohere
        response = co.generate(
            model="command-xlarge-beta",
            prompt=f"System: {system_content}\nUser: {prompt}",
            max_tokens=2000,
            temperature=0.7
        )
        reply = response.generations[0].text.strip()
        return JSONResponse(status_code=200, content={"reply": reply})

    except Exception as e:
        logging.exception("Chat Two follow-up failed")
        raise HTTPException(status_code=500, detail=str(e))