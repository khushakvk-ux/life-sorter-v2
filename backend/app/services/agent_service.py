"""
═══════════════════════════════════════════════════════════════
AI AGENT SERVICE — Dynamic Persona-Driven Question Generation
═══════════════════════════════════════════════════════════════
Uses OpenAI with loaded persona context to:
  1. Generate 2-3 highly relevant follow-up questions after Q1-Q3
  2. Produce personalized tool recommendations based on all answers
  3. Dynamically switch persona based on the domain selected

The agent loads the domain-specific persona document, parses it by task,
and extracts the Problems, Opportunities, Strategies and RCA Bridge
sections for the user's selected task. These structured sections are
used as deep context so follow-up questions directly reference the
documented problems and diagnostic signals from the persona docs.
"""

import json
from typing import Optional

import structlog
from openai import AsyncOpenAI

from app.config import get_settings
from app.services.persona_doc_service import load_persona_doc, load_task_context

logger = structlog.get_logger()


def _get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key_active)


# ── Dynamic Question Generation ────────────────────────────────


QUESTION_GENERATION_SYSTEM_PROMPT = """You are an expert AI business consultant running a diagnostic interview.

You have been provided with STRUCTURED DOMAIN CONTEXT directly from an authoritative persona document. This context contains:
- **PROBLEMS**: The specific, documented problems users face for this task
- **OPPORTUNITIES**: Concrete improvements and best practices available
- **STRATEGIES**: Proven strategic frameworks to address the problems
- **RCA BRIDGE**: Root-cause-analysis diagnostic signals (symptom → metric → root area)

Your job: Generate exactly {num_questions} diagnostic follow-up questions that will help you pinpoint WHICH of the documented problems the user is experiencing, and which opportunities/strategies are most relevant to their specific situation.

CRITICAL RULES:
- Each question MUST directly reference or probe a specific problem, opportunity, or strategy from the provided context — do NOT invent generic questions
- Use the documented PROBLEMS as inspiration: turn each problem or cluster of related problems into a diagnostic question
- Use the RCA BRIDGE signals: these are the exact symptoms and metrics you should probe for
- Options should map to different documented problems, current maturity levels, or readiness for opportunities
- Each question should have 3-5 multiple choice options covering common scenarios described in the documents
- Include one option allowing a different/custom answer
- Questions should be practical and conversational — not academic or survey-like
- DO NOT ask about their goal, domain, or task — they already answered those

OUTPUT FORMAT (strict JSON):
{{
  "questions": [
    {{
      "question": "Your question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "allows_free_text": true
    }}
  ]
}}

Return ONLY valid JSON, no markdown, no explanation."""


async def generate_dynamic_questions(
    outcome: str,
    outcome_label: str,
    domain: str,
    task: str,
    num_questions: int = 3,
) -> list[dict]:
    """
    Generate dynamic follow-up questions based on parsed persona task context.

    Loads the persona doc, parses the task-specific block (Problems, Opportunities,
    Strategies, RCA Bridge), and uses those sections to generate targeted diagnostic
    questions.
    """
    settings = get_settings()
    client = _get_client()

    # ── Load structured task context from persona doc ──────────
    task_ctx = load_task_context(domain, task)

    if task_ctx and task_ctx.get("problems"):
        # Build structured context from the parsed sections
        persona_context_parts = []
        persona_context_parts.append(f"MATCHED TASK: {task_ctx['task']}")

        if task_ctx.get("variants"):
            persona_context_parts.append(f"\nTASK VARIANTS:\n{task_ctx['variants']}")

        if task_ctx.get("adjacent_terms"):
            persona_context_parts.append(f"\nADJACENT TERMS (metrics/concepts to probe):\n{task_ctx['adjacent_terms']}")

        persona_context_parts.append(f"\nPROBLEMS (documented issues users face):\n{task_ctx['problems']}")

        if task_ctx.get("opportunities"):
            persona_context_parts.append(f"\nOPPORTUNITIES (improvements to assess readiness for):\n{task_ctx['opportunities']}")

        if task_ctx.get("strategies"):
            persona_context_parts.append(f"\nSTRATEGIES (frameworks to evaluate fit):\n{task_ctx['strategies']}")

        if task_ctx.get("rca_bridge"):
            persona_context_parts.append(f"\nRCA BRIDGE (symptom → metric → root area):\n{task_ctx['rca_bridge']}")

        persona_context = "\n".join(persona_context_parts)

        logger.info(
            "Using structured task context for question generation",
            domain=domain,
            task=task,
            matched_task=task_ctx["task"][:60],
            context_length=len(persona_context),
        )
    else:
        # Fallback to full doc if task parsing fails
        full_doc = load_persona_doc(domain)
        if full_doc:
            persona_context = full_doc
            logger.warning(
                "Task context parsing failed, using full doc as fallback",
                domain=domain,
                task=task,
            )
        else:
            logger.warning(
                "No persona doc found for domain, using generic context",
                domain=domain,
            )
            persona_context = f"Domain: {domain}. Task: {task}. Outcome: {outcome_label}."

    # Build the user message
    user_message = f"""USER'S SELECTIONS:
- Growth Goal: {outcome_label}
- Domain: {domain}
- Task: {task}

STRUCTURED DOMAIN CONTEXT (from authoritative persona document):
{persona_context}

Based on the PROBLEMS, OPPORTUNITIES, STRATEGIES, and RCA BRIDGE above, generate {num_questions} diagnostic questions that help identify:
1. Which specific problems from the document the user is currently experiencing
2. Their current maturity level relative to the documented opportunities
3. Which strategies would be most relevant to their situation

Each question should directly map to documented content — not generic consulting questions."""

    system_prompt = QUESTION_GENERATION_SYSTEM_PROMPT.format(
        num_questions=num_questions
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])

        logger.info(
            "Dynamic questions generated from persona context",
            domain=domain,
            task=task,
            count=len(questions),
        )

        return questions

    except json.JSONDecodeError as e:
        logger.error("Failed to parse dynamic questions JSON", error=str(e))
        return _fallback_questions(domain, task, task_ctx)
    except Exception as e:
        logger.error("Failed to generate dynamic questions", error=str(e))
        return _fallback_questions(domain, task, task_ctx)


def _fallback_questions(domain: str, task: str, task_ctx: Optional[dict] = None) -> list[dict]:
    """
    Fallback questions if AI generation fails.
    Uses the parsed task context to build relevant questions from the doc itself.
    """
    questions = []

    if task_ctx and task_ctx.get("problems"):
        # Extract first few problems and turn them into a diagnostic question
        problems_text = task_ctx["problems"]
        problem_lines = [
            line.strip() for line in problems_text.split("\n")
            if line.strip() and len(line.strip()) > 20
        ]

        if problem_lines:
            options = [p[:120] for p in problem_lines[:5]]
            questions.append({
                "question": f"Which of these documented problems best describes your current challenge with {task.lower()}?",
                "options": options,
                "allows_free_text": True,
            })

    if task_ctx and task_ctx.get("rca_bridge"):
        rca_text = task_ctx["rca_bridge"]
        rca_lines = [
            line.strip() for line in rca_text.split("\n")
            if line.strip() and len(line.strip()) > 15
        ]
        if rca_lines:
            options = [r[:120] for r in rca_lines[:5]]
            questions.append({
                "question": "Which symptom are you seeing most often?",
                "options": options,
                "allows_free_text": True,
            })

    # Generic fallback if nothing from docs
    if not questions:
        questions = [
            {
                "question": f"What tools or processes are you currently using for {task.lower()}?",
                "options": [
                    "Manual processes only",
                    "Basic tools (spreadsheets, email)",
                    "Some specialized software",
                    "Advanced/enterprise tools",
                ],
                "allows_free_text": True,
            },
            {
                "question": "What's your team size working on this?",
                "options": [
                    "Just me (solopreneur)",
                    "Small team (2-5 people)",
                    "Medium team (6-20 people)",
                    "Large team (20+ people)",
                ],
                "allows_free_text": True,
            },
            {
                "question": "What's your primary goal in the next 30 days?",
                "options": [
                    "Quick wins - start seeing results fast",
                    "Build a sustainable system/process",
                    "Scale what's already working",
                    "Fix something that's broken",
                ],
                "allows_free_text": True,
            },
        ]

    return questions


# ── Personalized Recommendation Generation ─────────────────────


RECOMMENDATION_SYSTEM_PROMPT = """You are an expert AI tools consultant. Based on the user's complete profile (their goals, domain, task, and answers to follow-up questions), recommend the BEST AI tools for their specific situation.

You have domain expertise context (PERSONA CONTEXT) to inform your recommendations.

For each recommendation, explain WHY it's specifically good for this user's situation based on their answers.

OUTPUT FORMAT (strict JSON):
{{
  "extensions": [
    {{
      "name": "Tool Name",
      "description": "What it does",
      "url": "https://...",
      "free": true,
      "why_recommended": "Specific reason based on user's answers"
    }}
  ],
  "gpts": [
    {{
      "name": "GPT Name",
      "description": "What it does",
      "url": "https://chat.openai.com/g/...",
      "rating": "4.8",
      "why_recommended": "Specific reason based on user's answers"
    }}
  ],
  "companies": [
    {{
      "name": "Company Name",
      "description": "What they do",
      "url": "https://...",
      "why_recommended": "Specific reason based on user's answers"
    }}
  ],
  "summary": "A 2-3 sentence personalized summary of why these tools were recommended"
}}

RULES:
- Recommend 2-4 items per category
- Every recommendation must have a specific 'why_recommended' tied to the user's answers
- Prioritize free/freemium tools when the user seems budget-conscious
- Prioritize ease of use for solopreneurs or small teams
- Prioritize scalability for larger teams
- Be specific and actionable
- Only recommend REAL tools that actually exist

Return ONLY valid JSON."""


async def generate_personalized_recommendations(
    outcome: str,
    outcome_label: str,
    domain: str,
    task: str,
    questions_answers: list[dict],
) -> dict:
    """
    Generate personalized tool recommendations based on all Q&A.

    Args:
        outcome: The outcome ID
        outcome_label: The outcome display label
        domain: The domain/sub-category
        task: The specific task
        questions_answers: List of all Q&A pairs (static + dynamic)

    Returns:
        Dict with 'extensions', 'gpts', 'companies', 'summary'
    """
    settings = get_settings()
    client = _get_client()

    # Load structured task context from persona doc
    task_ctx = load_task_context(domain, task)
    if task_ctx and task_ctx.get("problems"):
        # Build focused context from the matching task block
        context_parts = [f"MATCHED TASK: {task_ctx['task']}"]
        if task_ctx.get("problems"):
            context_parts.append(f"\nDOCUMENTED PROBLEMS:\n{task_ctx['problems']}")
        if task_ctx.get("opportunities"):
            context_parts.append(f"\nDOCUMENTED OPPORTUNITIES:\n{task_ctx['opportunities']}")
        if task_ctx.get("strategies"):
            context_parts.append(f"\nDOCUMENTED STRATEGIES:\n{task_ctx['strategies']}")
        persona_context = "\n".join(context_parts)
    else:
        full_doc = load_persona_doc(domain)
        persona_context = full_doc or f"Domain: {domain}. Task: {task}."

    # Build Q&A summary
    qa_text = ""
    for i, qa in enumerate(questions_answers, 1):
        qa_text += f"Q{i} ({qa.get('type', 'static')}): {qa.get('q', qa.get('question', ''))}\n"
        qa_text += f"A{i}: {qa.get('a', qa.get('answer', ''))}\n\n"

    user_message = f"""USER PROFILE:
- Growth Goal: {outcome_label}
- Domain: {domain}
- Task: {task}

ALL QUESTIONS & ANSWERS:
{qa_text}

PERSONA CONTEXT (domain expertise):
{persona_context}

Based on everything above, recommend the most relevant AI tools, Chrome extensions, Custom GPTs, and AI companies for this user's specific situation."""

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        logger.info(
            "Personalized recommendations generated",
            domain=domain,
            task=task,
            extensions=len(parsed.get("extensions", [])),
            gpts=len(parsed.get("gpts", [])),
            companies=len(parsed.get("companies", [])),
        )

        return {
            "extensions": parsed.get("extensions", []),
            "gpts": parsed.get("gpts", []),
            "companies": parsed.get("companies", []),
            "summary": parsed.get("summary", ""),
        }

    except json.JSONDecodeError as e:
        logger.error("Failed to parse recommendations JSON", error=str(e))
        return {"extensions": [], "gpts": [], "companies": [], "summary": ""}
    except Exception as e:
        logger.error("Failed to generate recommendations", error=str(e))
        return {"extensions": [], "gpts": [], "companies": [], "summary": ""}
