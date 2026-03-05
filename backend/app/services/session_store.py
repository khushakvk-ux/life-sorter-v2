"""
═══════════════════════════════════════════════════════════════
SESSION STORE — In-memory session context management
═══════════════════════════════════════════════════════════════
Stores and manages chat session contexts in memory.
Each session preserves the full flow: Q1-Q3, dynamic questions,
answers, persona context, and recommendations.

NOTE: In production, replace this with Redis or a database.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import structlog

from app.models.session import QuestionAnswer, SessionContext, SessionStage

logger = structlog.get_logger()

# In-memory session store (replace with Redis/DB in production)
_sessions: dict[str, SessionContext] = {}

# Max sessions to keep in memory (LRU eviction)
MAX_SESSIONS = 1000


def create_session() -> SessionContext:
    """Create a new session with a unique ID."""
    session_id = str(uuid.uuid4())
    session = SessionContext(session_id=session_id)

    # Evict oldest sessions if at capacity
    if len(_sessions) >= MAX_SESSIONS:
        oldest_id = min(_sessions, key=lambda k: _sessions[k].created_at)
        del _sessions[oldest_id]
        logger.info("Evicted oldest session", session_id=oldest_id)

    _sessions[session_id] = session
    logger.info("Session created", session_id=session_id)
    return session


def get_session(session_id: str) -> Optional[SessionContext]:
    """Retrieve a session by ID."""
    return _sessions.get(session_id)


def update_session(session: SessionContext) -> SessionContext:
    """Update a session in the store."""
    session.updated_at = datetime.utcnow()
    _sessions[session.session_id] = session
    return session


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def set_outcome(session_id: str, outcome: str, outcome_label: str) -> Optional[SessionContext]:
    """Set the Q1 answer (outcome/growth bucket)."""
    session = get_session(session_id)
    if not session:
        return None

    session.outcome = outcome
    session.outcome_label = outcome_label
    session.stage = SessionStage.DOMAIN
    session.questions_answers.append(
        QuestionAnswer(
            question="What matters most to you right now?",
            answer=outcome_label,
            question_type="static",
        )
    )
    return update_session(session)


def set_domain(session_id: str, domain: str) -> Optional[SessionContext]:
    """Set the Q2 answer (domain/sub-category)."""
    session = get_session(session_id)
    if not session:
        return None

    session.domain = domain
    session.stage = SessionStage.TASK
    session.questions_answers.append(
        QuestionAnswer(
            question="Which domain best matches your need?",
            answer=domain,
            question_type="static",
        )
    )
    return update_session(session)


def set_task(session_id: str, task: str) -> Optional[SessionContext]:
    """Set the Q3 answer (task). Moves to dynamic questions stage."""
    session = get_session(session_id)
    if not session:
        return None

    session.task = task
    session.stage = SessionStage.DYNAMIC_QUESTIONS
    session.questions_answers.append(
        QuestionAnswer(
            question="What task would you like help with?",
            answer=task,
            question_type="static",
        )
    )
    return update_session(session)


def add_dynamic_answer(
    session_id: str, question: str, answer: str
) -> Optional[SessionContext]:
    """Record a dynamic question answer."""
    session = get_session(session_id)
    if not session:
        return None

    session.questions_answers.append(
        QuestionAnswer(
            question=question,
            answer=answer,
            question_type="dynamic",
        )
    )
    session.dynamic_questions_asked += 1

    # If all dynamic questions answered, move to recommendation
    if session.dynamic_questions_asked >= session.dynamic_questions_total:
        session.stage = SessionStage.RECOMMENDATION

    return update_session(session)


def set_recommendations(
    session_id: str,
    extensions: list[dict],
    gpts: list[dict],
    companies: list[dict],
) -> Optional[SessionContext]:
    """Store the final recommendations in the session."""
    session = get_session(session_id)
    if not session:
        return None

    session.recommended_extensions = extensions
    session.recommended_gpts = gpts
    session.recommended_companies = companies
    session.stage = SessionStage.COMPLETE
    return update_session(session)


# ── Claude RCA helpers ─────────────────────────────────────────

def set_rca_context(
    session_id: str, diagnostic_context: dict
) -> Optional[SessionContext]:
    """Store the raw dynamic-loader output as internal RCA context."""
    session = get_session(session_id)
    if not session:
        return None
    session.rca_diagnostic_context = diagnostic_context
    return update_session(session)


def add_rca_answer(
    session_id: str, question: str, answer: str
) -> Optional[SessionContext]:
    """Append a Claude RCA question-answer pair."""
    session = get_session(session_id)
    if not session:
        return None
    session.rca_history.append({"question": question, "answer": answer})
    session.questions_answers.append(
        QuestionAnswer(question=question, answer=answer, question_type="rca")
    )
    return update_session(session)


def set_rca_complete(
    session_id: str, summary: str = ""
) -> Optional[SessionContext]:
    """Mark the RCA diagnostic as complete."""
    session = get_session(session_id)
    if not session:
        return None
    session.rca_complete = True
    session.rca_summary = summary
    session.stage = SessionStage.RECOMMENDATION
    return update_session(session)


def set_rca_fallback(session_id: str) -> Optional[SessionContext]:
    """Activate fallback mode (use static dynamic-loader questions)."""
    session = get_session(session_id)
    if not session:
        return None
    session.rca_fallback_active = True
    return update_session(session)


# ── Early Recommendations helpers ──────────────────────────────

def set_early_recommendations(
    session_id: str,
    tools: list[dict],
    message: str = "",
) -> Optional[SessionContext]:
    """Store early tool recommendations generated after Q3."""
    session = get_session(session_id)
    if not session:
        return None
    session.early_recommendations = tools
    session.early_recommendations_message = message
    return update_session(session)


# ── Website & Audience Insights helpers ────────────────────────

def set_website_url(
    session_id: str, website_url: str, url_type: str = "website"
) -> Optional[SessionContext]:
    """Store the user's business website URL with metadata."""
    session = get_session(session_id)
    if not session:
        return None
    session.website_url = website_url
    session.url_type = url_type
    session.url_submitted_at = datetime.utcnow().isoformat() + "Z"
    return update_session(session)


def set_audience_insights(
    session_id: str, insights: dict
) -> Optional[SessionContext]:
    """Store audience analysis insights from website review."""
    session = get_session(session_id)
    if not session:
        return None
    session.audience_insights = insights
    return update_session(session)


def set_crawl_status(
    session_id: str, status: str
) -> Optional[SessionContext]:
    """Update the crawl status flag (in_progress, complete, failed)."""
    session = get_session(session_id)
    if not session:
        return None
    session.crawl_status = status
    return update_session(session)


def set_crawl_data(
    session_id: str,
    crawl_raw: dict,
    crawl_summary: dict,
) -> Optional[SessionContext]:
    """Store both raw crawl data and the compressed summary."""
    session = get_session(session_id)
    if not session:
        return None
    session.crawl_raw = crawl_raw
    session.crawl_summary = crawl_summary
    session.crawl_status = crawl_summary.get("crawl_status", "complete")
    return update_session(session)


# ── Business Profile / Scale Questions helpers ─────────────────

def set_business_profile(
    session_id: str, profile: dict
) -> Optional[SessionContext]:
    """Store the business profile from scale questions."""
    session = get_session(session_id)
    if not session:
        return None
    session.business_profile = profile
    session.scale_questions_complete = True
    session.stage = SessionStage.DYNAMIC_QUESTIONS
    # Also record each scale answer in the main Q&A list for traceability
    for key, value in profile.items():
        session.questions_answers.append(
            QuestionAnswer(
                question=f"Scale: {key}",
                answer=str(value),
                question_type="scale",
            )
        )
    return update_session(session)


def get_session_summary(session_id: str) -> Optional[dict]:
    """Get a summary of the full session context."""
    session = get_session(session_id)
    if not session:
        return None

    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "stage": session.stage.value,
        "outcome": session.outcome_label,
        "domain": session.domain,
        "task": session.task,
        "persona_doc": session.persona_doc_name,
        "questions_answers": [
            {"q": qa.question, "a": qa.answer, "type": qa.question_type}
            for qa in session.questions_answers
        ],
        "dynamic_questions_progress": f"{session.dynamic_questions_asked}/{session.dynamic_questions_total}",
        "recommendations": {
            "extensions": len(session.recommended_extensions),
            "gpts": len(session.recommended_gpts),
            "companies": len(session.recommended_companies),
        },
        "website_url": session.website_url,
        "url_type": session.url_type,
        "audience_insights": session.audience_insights if session.audience_insights else None,
        "crawl_status": session.crawl_status or None,
        "crawl_summary": session.crawl_summary if session.crawl_summary else None,
        "business_profile": session.business_profile if session.business_profile else None,
    }
