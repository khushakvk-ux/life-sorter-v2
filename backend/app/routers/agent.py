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
from app.services.claude_rca_service import generate_next_rca_question
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
    rca_mode: bool = False          # True = Claude adaptive, False = static fallback
    acknowledgment: str = ""        # Claude's acknowledgment text (first question only)


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
    1. Loads diagnostic context from persona docs (internal only).
    2. Calls Claude via OpenRouter for the FIRST adaptive RCA question.
    3. Falls back to static persona-doc questions if Claude fails.
    """
    session = session_store.set_task(body.session_id, body.task)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Look up which persona doc this domain maps to
    persona_doc_name = get_doc_for_domain(session.domain or "")
    session.persona_doc_name = persona_doc_name
    session.persona_context_loaded = persona_doc_name is not None

    # Load diagnostic context from pre-parsed document (used as internal context)
    diagnostic = get_diagnostic_sections(
        domain=session.domain or "",
        task=session.task or "",
    )

    task_matched = ""
    if diagnostic:
        task_matched = diagnostic.get("task_matched", "")
        # Store the full diagnostic context for Claude to use
        session_store.set_rca_context(session.session_id, diagnostic)

    # ── Try Claude RCA for the first question ──────────────────
    claude_result = await generate_next_rca_question(
        outcome=session.outcome or "",
        outcome_label=session.outcome_label or "",
        domain=session.domain or "",
        task=session.task or "",
        diagnostic_context=diagnostic or {},
        rca_history=[],
    )

    if claude_result and claude_result.get("status") == "question":
        # Claude gave us the first adaptive question
        first_q = DynamicQuestion(
            question=claude_result["question"],
            options=claude_result.get("options", []),
            allows_free_text=True,
            section=claude_result.get("section", "rca"),
            section_label=claude_result.get("section_label", "Diagnostic"),
        )

        # Store question text for tracking
        session.dynamic_questions = [claude_result["question"]]
        session.dynamic_questions_total = -1  # Unknown — Claude decides
        session_store.update_session(session)

        logger.info(
            "Claude RCA: first question generated",
            session_id=session.session_id,
            question=claude_result["question"][:80],
        )

        return SetTaskResponse(
            session_id=session.session_id,
            stage=session.stage.value,
            persona_loaded=persona_doc_name or "generic",
            task_matched=task_matched,
            questions=[first_q],  # Single question — frontend handles adaptively
            rca_mode=True,
            acknowledgment=claude_result.get("acknowledgment", ""),
        )

    # ── Fallback: static persona-doc questions ─────────────────
    logger.warning(
        "Claude RCA unavailable, falling back to static questions",
        session_id=session.session_id,
    )
    session_store.set_rca_fallback(session.session_id)

    dynamic_qs = []
    if diagnostic and diagnostic.get("sections"):
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
        "Fallback: static diagnostic sections loaded",
        session_id=session.session_id,
        num_sections=len(dynamic_qs),
    )

    return SetTaskResponse(
        session_id=session.session_id,
        stage=session.stage.value,
        persona_loaded=persona_doc_name or "generic",
        task_matched=task_matched,
        questions=dynamic_qs,
        rca_mode=False,
    )


@router.post("/session/answer", response_model=SubmitDynamicAnswerResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT_CHAT)
async def submit_dynamic_answer(request: Request, body: SubmitDynamicAnswerRequest = Body(...)):
    """
    Submit an answer to a diagnostic question.
    In RCA mode: sends all context + history to Claude → gets next adaptive question.
    In fallback mode: advances through static question list.
    """
    session = session_store.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── RCA Mode (Claude adaptive) ─────────────────────────────
    if not session.rca_fallback_active:
        # Record this answer
        question_text = (
            session.dynamic_questions[body.question_index]
            if body.question_index < len(session.dynamic_questions)
            else f"RCA Question {body.question_index + 1}"
        )
        session_store.add_rca_answer(
            body.session_id, question_text, body.answer
        )

        # Refresh session after adding answer
        session = session_store.get_session(body.session_id)

        # Ask Claude for the next question
        claude_result = await generate_next_rca_question(
            outcome=session.outcome or "",
            outcome_label=session.outcome_label or "",
            domain=session.domain or "",
            task=session.task or "",
            diagnostic_context=session.rca_diagnostic_context,
            rca_history=session.rca_history,
        )

        if claude_result and claude_result.get("status") == "question":
            next_q = DynamicQuestion(
                question=claude_result["question"],
                options=claude_result.get("options", []),
                allows_free_text=True,
                section=claude_result.get("section", "rca"),
                section_label=claude_result.get("section_label", "Diagnostic"),
            )
            # Track the question text
            session.dynamic_questions.append(claude_result["question"])
            session_store.update_session(session)

            logger.info(
                "Claude RCA: next question",
                session_id=session.session_id,
                q_index=len(session.rca_history),
                question=claude_result["question"][:80],
            )

            return SubmitDynamicAnswerResponse(
                session_id=session.session_id,
                next_question=next_q,
                all_answered=False,
                rca_mode=True,
                acknowledgment=claude_result.get("acknowledgment", ""),
            )

        elif claude_result and claude_result.get("status") == "complete":
            # Claude says we have enough — move to recommendation
            summary = claude_result.get("summary", "")
            session_store.set_rca_complete(body.session_id, summary)

            logger.info(
                "Claude RCA: diagnostic complete",
                session_id=session.session_id,
                total_questions=len(session.rca_history),
                summary=summary[:100],
            )

            return SubmitDynamicAnswerResponse(
                session_id=session.session_id,
                next_question=None,
                all_answered=True,
                rca_mode=True,
                acknowledgment=claude_result.get("acknowledgment", ""),
                rca_summary=summary,
            )

        else:
            # Claude failed mid-flow — mark as complete and move on
            logger.warning(
                "Claude RCA failed mid-flow, completing diagnostic",
                session_id=session.session_id,
            )
            session_store.set_rca_complete(body.session_id, "")
            return SubmitDynamicAnswerResponse(
                session_id=session.session_id,
                next_question=None,
                all_answered=True,
                rca_mode=True,
            )

    # ── Fallback Mode (static questions) ───────────────────────
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
        next_question = DynamicQuestion(
            question=session.dynamic_questions[next_index],
            options=[],
            allows_free_text=True,
        )

    return SubmitDynamicAnswerResponse(
        session_id=session.session_id,
        next_question=next_question,
        all_answered=all_answered,
        rca_mode=False,
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
