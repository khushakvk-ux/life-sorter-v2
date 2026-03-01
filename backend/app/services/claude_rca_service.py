"""
═══════════════════════════════════════════════════════════════
CLAUDE RCA SERVICE — Root Cause Analysis via Claude Opus 4.6
═══════════════════════════════════════════════════════════════
Calls Claude Sonnet via the OpenRouter API to generate
adaptive, layman-friendly diagnostic questions one at a time.

Takes dynamic-loader context (problems, RCA bridge symptoms,
opportunities, strategies) + user's Q1-Q3 answers + previous
RCA Q&A history → returns the next question or signals "done".

Fallback: if Claude is unreachable, the old dynamic-loader
questions are served directly (pre-parsed from persona docs).
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
import structlog
import json

from app.config import get_settings

logger = structlog.get_logger()

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Ikshan — a world-class business diagnostic advisor powered by deep \
domain intelligence. You diagnose business bottlenecks the way a ₹50-lakh/year \
consultant would, but in plain, friendly language anyone can follow.

The user has already told us three things:
• Q1 — the business outcome they care about most
• Q2 — the specific domain they work in
• Q3 — the exact task they need help with

You now receive rich, expert-curated context from our knowledge base:
  – Real-world **problem patterns** seen in this task
  – **Diagnostic signals** (symptom → KPI/metric → root-cause area)
  – **Growth opportunities** specific to this task
  – **Proven strategies & frameworks** that top operators use
  – **RCA bridge data** mapping visible symptoms to hidden root causes

Your job: use ALL of this intelligence to conduct a diagnostic that makes the \
user think "Wow, this tool really understands my problem."

═══ HOW TO ASK IMPRESSIVE QUESTIONS ═══

1. **Show domain expertise in every question.** Don't ask generic "what's your \
   problem?" — reference specific patterns from the knowledge base.
   BAD:  "What's not working with your social media?"
   GOOD: "When your posts get views but nobody comments, that usually means \
          the hook grabbed attention but the caption didn't give them a reason \
          to respond. Which of these sounds closer to your situation?"

2. **Weave in real metrics and benchmarks.** Use the diagnostic signals data \
   to reference KPIs that matter. This makes the user feel you speak their \
   language even while keeping it simple.
   Example: "Most businesses in your space see 2-4% engagement rates. Posts \
            that convert to leads typically need a clear next step. Looking at \
            your content, where does the drop-off happen?"

3. **Explain the 'why' briefly.** Before each question, add a 1-2 sentence \
   insight that educates the user and shows WHY this question matters.
   Example: "Here's what I've seen across hundreds of businesses like yours: \
            the #1 reason social posts get views but no leads is a missing \
            bridge between the content and the next step. Let me narrow it \
            down for you…"

4. **Use the RCA bridge mapping.** When you know a symptom maps to a specific \
   root-cause area (e.g., "Execution/Production" or "QA/Review/Controls"), \
   craft questions that probe that area without using jargon.

5. **Options should be specific and recognizable.** Each option should describe \
   a concrete, real scenario the user can immediately relate to — not vague \
   categories. Pull directly from the problems and symptoms data.

6. **Progressive depth.** Start with the visible pain, then drill into the \
   underlying cause, then uncover the systemic gap. The question flow should \
   feel like peeling layers of an onion, not random sampling.
   - RCA-Q1: Identify the visible symptom / pain point
   - RCA-Q2: Probe what's behind it (root behavior / process gap)
   - RCA-Q3: Uncover the systemic gap (missing system / framework)
   - RCA-Q4: Validate your understanding with a sharper follow-up
   - RCA-Q5: Confirm priority and readiness (power-move question)

7. **End with a power move.** Your final question (or the one before completion) \
   should give the user a moment of clarity — an "aha" where they realize \
   the root cause themselves. This is what makes them want deeper analysis.

═══ TONE & STYLE ═══

- Talk like a smart, caring advisor at a coffee shop — warm but incisive.
- Use "I" and "you" — it's a conversation, not a form.
- Include brief analogies or relatable comparisons when helpful.
- Show genuine curiosity about their specific situation.
- Each acknowledgment should contain a micro-insight, not just empathy.
  BAD:  "That makes sense."
  GOOD: "That's actually one of the top 3 patterns I see — when hooks don't \
         stop the scroll, it usually means the opening words aren't hitting a \
         nerve the reader cares about right now."

═══ RESPONSE RULES ═══

1. ONE question per response. No exceptions.
2. 3-6 answer options per question. Always include "Something else" as the last.
3. You must ask a MINIMUM of 4 diagnostic questions. Aim for 5. \
   Do NOT signal "complete" before asking at least 4 questions. \
   After 6 questions you MUST signal completion. Absolute max is 7. \
   The user's earlier Q1 (outcome), Q2 (domain), Q3 (task) do NOT count — \
   your count starts from YOUR first diagnostic question.
4. Every option must be a specific, recognizable scenario (not generic labels).
5. The question text should be 2-4 sentences: micro-insight + the actual question.
6. Acknowledgments must contain a useful observation, not just validation.

═══ RESPONSE FORMAT ═══

Respond in valid JSON only:

When asking a question:
{
  "status": "question",
  "acknowledgment": "1-2 sentence micro-insight acknowledging their answer",
  "question": "2-4 sentence question with context insight + the actual question",
  "options": ["Specific scenario A", "Specific scenario B", "Specific scenario C", "Something else"],
  "section": "problems|rca_bridge|opportunities|deepdive",
  "section_label": "Crisp, specific label (e.g., 'Hook & Scroll-Stop Analysis')"
}

When diagnostic is complete:
{
  "status": "complete",
  "acknowledgment": "1-2 sentence power insight that gives them an 'aha' moment",
  "summary": "3-4 sentence summary: what's broken, why it's broken, and the specific root cause area. Make them feel understood and eager for the full analysis."
}
"""


def _build_user_context(
    outcome: str,
    outcome_label: str,
    domain: str,
    task: str,
    diagnostic_context: dict[str, Any],
    rca_history: list[dict[str, str]],
) -> str:
    """Build the rich user-context message sent alongside the system prompt."""
    parts = [
        "═══ USER PROFILE ═══",
        f"Outcome they want (Q1): {outcome_label}",
        f"Business domain  (Q2): {domain}",
        f"Specific task    (Q3): {task}",
    ]

    if not diagnostic_context:
        parts.append("\n(No domain-specific context available — use your general knowledge.)")
    else:
        matched_task = diagnostic_context.get("task_matched", "")
        if matched_task:
            parts.append(f"\nMatched knowledge-base task: \"{matched_task}\"")

        # ── Full context from parsed doc ───────────────────────
        full_ctx = diagnostic_context.get("full_context", {})

        # Variants — shows related phrasings of this task
        variants = full_ctx.get("variants", "")
        if variants:
            parts.append(f"\n═══ TASK VARIANTS (how people describe this task) ═══")
            parts.append(variants[:400])

        # ── Section: Real-world Problems ───────────────────────
        sections = diagnostic_context.get("sections", [])
        for sec in sections:
            key = sec.get("key", "")
            label = sec.get("label", key)
            items = sec.get("items", [])

            if key == "problems":
                parts.append(f"\n═══ REAL-WORLD PROBLEM PATTERNS (use these to craft precise options) ═══")
                for i, item in enumerate(items, 1):
                    parts.append(f"  P{i}. {item}")

            elif key == "rca_bridge":
                parts.append(f"\n═══ DIAGNOSTIC SIGNALS (symptom → metric → root cause area) ═══")
                parts.append("Use these to show domain expertise. Reference metrics naturally.")
                rca_parsed = sec.get("rca_parsed", [])
                if rca_parsed:
                    for i, rca in enumerate(rca_parsed, 1):
                        sym = rca.get("symptom", "")
                        met = rca.get("metric", "")
                        root = rca.get("root_area", "")
                        line = f"  S{i}. \"{sym}\""
                        if met:
                            line += f" → KPI: {met}"
                        if root:
                            line += f" → Root: {root}"
                        parts.append(line)
                else:
                    for i, item in enumerate(items, 1):
                        parts.append(f"  S{i}. {item}")

            elif key == "opportunities":
                parts.append(f"\n═══ GROWTH OPPORTUNITIES (what good looks like) ═══")
                for i, item in enumerate(items, 1):
                    parts.append(f"  O{i}. {item}")

        # ── Strategies & frameworks ────────────────────────────
        strategies = diagnostic_context.get("strategies", "")
        if strategies:
            parts.append(f"\n═══ PROVEN STRATEGIES & FRAMEWORKS ═══")
            parts.append("Reference these naturally to show expertise (don't dump them on the user).")
            parts.append(strategies[:2000])

    # ── Previous Q&A history ───────────────────────────────────
    if rca_history:
        parts.append(f"\n═══ DIAGNOSTIC CONVERSATION SO FAR ({len(rca_history)} of your questions asked) ═══")
        for i, qa in enumerate(rca_history, 1):
            parts.append(f"  RCA-Q{i}: {qa['question']}")
            parts.append(f"  RCA-A{i}: {qa['answer']}")
        remaining = max(0, 4 - len(rca_history))
        if remaining > 0:
            parts.append(
                f"\n→ You have asked {len(rca_history)} diagnostic question(s). "
                f"You MUST ask at least {remaining} more before you can signal 'complete'. "
                "Generate the NEXT question. Build on their answers. Drill deeper."
            )
        else:
            parts.append(
                "\n→ Generate the NEXT question. Build on their answers. "
                "Drill deeper into the root cause. If you have enough info "
                "you may signal 'complete', or ask one more to sharpen the diagnosis."
            )
    else:
        parts.append(
            "\n→ This is the FIRST question. Start by identifying the most "
            "pressing visible problem. Use a micro-insight from the knowledge "
            "base to show you understand their world before asking."
        )

    return "\n".join(parts)


async def generate_next_rca_question(
    outcome: str,
    outcome_label: str,
    domain: str,
    task: str,
    diagnostic_context: dict[str, Any],
    rca_history: list[dict[str, str]],
) -> Optional[dict[str, Any]]:
    """
    Call Claude via OpenRouter to get the next adaptive RCA question.

    Returns a dict with either:
      {"status": "question", "question": ..., "options": [...], ...}
      {"status": "complete", "summary": ...}
      None on failure (caller should fall back to static questions)
    """
    settings = get_settings()
    api_key = settings.OPENROUTER_API_KEY
    model = settings.OPENROUTER_MODEL

    if not api_key:
        logger.warning("OpenRouter API key not configured — falling back")
        return None

    user_content = _build_user_context(
        outcome, outcome_label, domain, task, diagnostic_context, rca_history
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ikshan.ai",
        "X-Title": "Ikshan RCA Engine",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENROUTER_CHAT_URL, json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)

        # Validate expected fields
        if result.get("status") not in ("question", "complete"):
            logger.error("Unexpected Claude response status", raw=content[:300])
            return None

        logger.info(
            "Claude RCA response",
            status=result["status"],
            question=result.get("question", "")[:80],
            num_options=len(result.get("options", [])),
        )
        return result

    except httpx.HTTPStatusError as e:
        logger.error(
            "OpenRouter HTTP error",
            status_code=e.response.status_code,
            body=e.response.text[:300],
        )
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("Claude response parse error", error=str(e))
        return None
    except httpx.RequestError as e:
        logger.error("OpenRouter request failed", error=str(e))
        return None
