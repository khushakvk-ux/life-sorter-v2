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
    }
