"""
═══════════════════════════════════════════════════════════════
AGENT ROUTER — AI Agent with Dynamic Persona & Session Context
═══════════════════════════════════════════════════════════════
Endpoints for the AI agent flow:

POST /api/v1/agent/session              — Create a new session
POST /api/v1/agent/session/outcome      — Record Q1 (outcome)
POST /api/v1/agent/session/domain       — Record Q2 (domain)
POST /api/v1/agent/session/task         — Record Q3 (task) + generate dynamic Qs
POST /api/v1/agent/session/answer       — Submit dynamic question answer
POST /api/v1/agent/session/recommend    — Get final personalized recommendations
GET  /api/v1/agent/session/{id}         — Get full session context
GET  /api/v1/agent/personas             — List available persona domains
"""

import structlog
from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Optional

from app.config import get_settings
from app.middleware.rate_limit import limiter
from app.services import session_store, agent_service
from app.services.persona_doc_service import get_available_personas, get_doc_for_domain, get_diagnostic_sections
from app.models.session import (
    SessionStage,
    GenerateDynamicQuestionsRequest,
    GenerateDynamicQuestionsResponse,
    DynamicQuestion,
    SubmitDynamicAnswerRequest,
    SubmitDynamicAnswerResponse,
    GetRecommendationsRequest,
    GetRecommendationsResponse,
    ToolRecommendation,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/agent", tags=["agent"])


# ── Request/Response Models ────────────────────────────────────


class CreateSessionResponse(BaseModel):
    session_id: str
    stage: str


class SetOutcomeRequest(BaseModel):
    session_id: str
    outcome: str           # e.g., 'lead-generation'
    outcome_label: str     # e.g., 'Lead Generation (Marketing, SEO & Social)'


class SetDomainRequest(BaseModel):
    session_id: str
    domain: str            # e.g., 'Content & Social Media'


class SetTaskRequest(BaseModel):
    session_id: str
    task: str              # e.g., 'Generate social media posts captions & hooks'


class SetTaskResponse(BaseModel):
    session_id: str
    stage: str
    persona_loaded: str
    task_matched: str = ""
    questions: list[DynamicQuestion]


class SessionContextResponse(BaseModel):
    session_id: str
    stage: str
    outcome: Optional[str] = None
    outcome_label: Optional[str] = None
    domain: Optional[str] = None
    task: Optional[str] = None
    persona_doc: Optional[str] = None
    questions_answers: list[dict[str, Any]] = []
    dynamic_questions_progress: str = "0/0"
    recommendations: dict[str, Any] = {}


# ── Endpoints ──────────────────────────────────────────────────


@router.post("/session", response_model=CreateSessionResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_DEFAULT)
async def create_session(request: Request):
    """Create a new chat session."""
    session = session_store.create_session()
    return CreateSessionResponse(
        session_id=session.session_id,
        stage=session.stage.value,
    )


@router.post("/session/outcome", response_model=CreateSessionResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_DEFAULT)
async def set_outcome(request: Request, body: SetOutcomeRequest = Body(...)):
    """Record Q1: Outcome / Growth Bucket selection."""
    session = session_store.set_outcome(
        body.session_id, body.outcome, body.outcome_label
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return CreateSessionResponse(
        session_id=session.session_id,
        stage=session.stage.value,
    )


@router.post("/session/domain", response_model=CreateSessionResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_DEFAULT)
async def set_domain(request: Request, body: SetDomainRequest = Body(...)):
    """Record Q2: Domain / Sub-Category selection."""
    session = session_store.set_domain(body.session_id, body.domain)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return CreateSessionResponse(
        session_id=session.session_id,
        stage=session.stage.value,
    )


@router.post("/session/task", response_model=SetTaskResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_CHAT)
async def set_task_and_generate_questions(request: Request, body: SetTaskRequest = Body(...)):
    """
    Record Q3: Task selection.
    Loads diagnostic sections directly from the persona document.
    No GPT call — content comes straight from the pre-parsed .docx files.
    Returns Problems, RCA Bridge symptoms, and Opportunities as structured questions.
    """
    session = session_store.set_task(body.session_id, body.task)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Look up which persona doc this domain maps to
    persona_doc_name = get_doc_for_domain(session.domain or "")
    session.persona_doc_name = persona_doc_name
    session.persona_context_loaded = persona_doc_name is not None

    # Load diagnostic sections from pre-parsed document (instant, no GPT)
    diagnostic = get_diagnostic_sections(
        domain=session.domain or "",
        task=session.task or "",
    )

    dynamic_qs = []
    task_matched = ""

    if diagnostic and diagnostic.get("sections"):
        task_matched = diagnostic.get("task_matched", "")
        for section in diagnostic["sections"]:
            dq = DynamicQuestion(
                question=section["question"],
                options=section["items"],
                allows_free_text=section.get("allows_free_text", True),
                section=section["key"],
                section_label=section["label"],
            )
            dynamic_qs.append(dq)
            session.dynamic_questions.append(section["question"])

    session.dynamic_questions_total = len(dynamic_qs)
    session_store.update_session(session)

    logger.info(
        "Task set, diagnostic sections loaded from document",
        session_id=session.session_id,
        domain=session.domain,
        task=session.task,
        task_matched=task_matched,
        num_sections=len(dynamic_qs),
        persona=persona_doc_name,
    )

    return SetTaskResponse(
        session_id=session.session_id,
        stage=session.stage.value,
        persona_loaded=persona_doc_name or "generic",
        task_matched=task_matched,
        questions=dynamic_qs,
    )


@router.post("/session/answer", response_model=SubmitDynamicAnswerResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_CHAT)
async def submit_dynamic_answer(request: Request, body: SubmitDynamicAnswerRequest = Body(...)):
    """Submit an answer to a dynamic question."""
    session = session_store.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if body.question_index >= len(session.dynamic_questions):
        raise HTTPException(status_code=400, detail="Invalid question index")

    question_text = session.dynamic_questions[body.question_index]

    session = session_store.add_dynamic_answer(
        body.session_id, question_text, body.answer
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Determine next question or if all done
    next_index = body.question_index + 1
    all_answered = next_index >= session.dynamic_questions_total

    next_question = None
    if not all_answered and next_index < len(session.dynamic_questions):
        # Return the next question from our stored list
        # We need to reconstruct the DynamicQuestion — we stored the text, 
        # but options are re-fetch from the original generation
        next_question = DynamicQuestion(
            question=session.dynamic_questions[next_index],
            options=[],  # Options were already sent to the frontend
            allows_free_text=True,
        )

    return SubmitDynamicAnswerResponse(
        session_id=session.session_id,
        next_question=next_question,
        all_answered=all_answered,
    )


@router.post("/session/recommend", response_model=GetRecommendationsResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_CHAT)
async def get_recommendations(request: Request, body: GetRecommendationsRequest = Body(...)):
    """
    Generate final personalized tool recommendations based on
    all Q&A (static Q1-Q3 + dynamic questions).
    """
    settings = get_settings()
    if not settings.openai_api_key_active:
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable — OpenAI API key not configured.",
        )

    session = session_store.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build Q&A list
    qa_list = [
        {"q": qa.question, "a": qa.answer, "type": qa.question_type}
        for qa in session.questions_answers
    ]

    # Generate personalized recommendations
    recs = await agent_service.generate_personalized_recommendations(
        outcome=session.outcome or "",
        outcome_label=session.outcome_label or "",
        domain=session.domain or "",
        task=session.task or "",
        questions_answers=qa_list,
    )

    # Store in session
    session_store.set_recommendations(
        session.session_id,
        extensions=recs.get("extensions", []),
        gpts=recs.get("gpts", []),
        companies=recs.get("companies", []),
    )

    # Build response
    extensions = [
        ToolRecommendation(
            name=ext.get("name", ""),
            description=ext.get("description", ""),
            url=ext.get("url"),
            category="extension",
            free=ext.get("free"),
            why_recommended=ext.get("why_recommended", ""),
        )
        for ext in recs.get("extensions", [])
    ]

    gpts = [
        ToolRecommendation(
            name=gpt.get("name", ""),
            description=gpt.get("description", ""),
            url=gpt.get("url"),
            category="gpt",
            rating=gpt.get("rating"),
            why_recommended=gpt.get("why_recommended", ""),
        )
        for gpt in recs.get("gpts", [])
    ]

    companies = [
        ToolRecommendation(
            name=co.get("name", ""),
            description=co.get("description", ""),
            url=co.get("url"),
            category="company",
            why_recommended=co.get("why_recommended", ""),
        )
        for co in recs.get("companies", [])
    ]

    # Get session summary for context
    summary = session_store.get_session_summary(session.session_id) or {}

    return GetRecommendationsResponse(
        session_id=session.session_id,
        extensions=extensions,
        gpts=gpts,
        companies=companies,
        summary=recs.get("summary", ""),
        session_context=summary,
    )


@router.get("/session/{session_id}", response_model=SessionContextResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_DEFAULT)
async def get_session_context(request: Request, session_id: str):
    """Get the full session context (for debugging or UI state recovery)."""
    summary = session_store.get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionContextResponse(**summary)


@router.get("/personas")
@limiter.limit(lambda: get_settings().RATE_LIMIT_DEFAULT)
async def list_personas(request: Request):
    """List all available persona domains with document mappings."""
    personas = get_available_personas()
    return {"personas": personas, "count": len(personas)}
