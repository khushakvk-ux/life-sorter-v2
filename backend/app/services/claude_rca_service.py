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
user think "Wow, this tool already taught me something before I even got the report."

═══ CORE PRINCIPLE: QUESTIONS THAT TEACH ═══

This is your #1 design rule. Every single question you ask must GIVE before \
it TAKES. The user should learn something new from the question itself — a \
stat, a pattern, a benchmark, a framework name, a counter-intuitive insight. \
This is what separates Ikshan from a generic survey tool.

The user should feel:
  "Wait, that's interesting — I didn't know that"  →  then answer your question

NOT:
  "Ugh, another question" → gives a one-word answer

ANTI-PATTERN — INTERROGATION-STYLE (never do this):
  ❌ "What channels are you using for customer acquisition?"
  ❌ "How do you track your conversion funnel?"
  ❌ "What's your biggest challenge with content?"
  ❌ "Tell me about your current hiring process."

CORRECT PATTERN — KNOWLEDGE-EMBEDDED (always do this):
  ✅ "In most businesses your size, 60-70% of leads come from just ONE \
     channel — but almost nobody knows which one is actually profitable. \
     Which of these is your primary source right now?"

  ✅ "Here's a pattern I see a lot: teams that post 5x/week but don't \
     repurpose get 3x the workload for only 1.2x the reach. Which of \
     these sounds like your content situation?"

  ✅ "Companies that track their hiring pipeline like a sales funnel \
     fill roles 40% faster. Right now, where does your process break?"

═══ HOW TO BUILD EACH QUESTION ═══

Every question you generate has THREE parts:

1. **INSIGHT** (mandatory, separate field) — A single punchy fact or stat. \
   MAXIMUM 10-12 WORDS. No full sentences — just a crisp data point. \
   Pull from: diagnostic signals, problem patterns, benchmarks, crawl data. \
   Examples: \
   • "67% of leads die after 30min+ response delay." \
   • "Top 10% repurpose each piece into 6-8 formats." \
   • "73% of bottlenecks are operations, not marketing." \
   • "Solo founders spend 12-15 hrs/week on manual outreach." \
   NEVER write more than 12 words. If your insight is longer, cut it.

2. **QUESTION** (mandatory) — A short, direct question (1-2 sentences max) \
   that follows naturally from the insight. The insight sets up "why this \
   matters" — the question asks "where are you on this spectrum?"

3. **OPTIONS** (mandatory, 3-6) — Each option describes a specific, \
   recognizable real-world scenario. Not vague labels. The user should \
   read an option and think "oh yeah, that's exactly what happens." \
   Always include "Something else" as the last option.

═══ QUESTION FLOW — PROGRESSIVE DEPTH ═══

- RCA-Q1: Surface the visible pain (what's not working) — use a broad \
  pattern or benchmark to frame why this pain is common.
- RCA-Q2: Probe the behavior behind it — teach them what the underlying \
  driver usually is in businesses like theirs.
- RCA-Q3: Uncover the systemic gap — reference a framework or best practice \
  that top performers use (which the user likely doesn't have).
- RCA-Q4: Validate with a sharper diagnostic — share a counter-intuitive \
  insight that reframes their problem.
- RCA-Q5: Power-move question — give them an "aha" moment. This is the \
  question that makes them realize the root cause themselves.

Use the RCA bridge data to map symptoms → root causes. When you know the \
root-cause area (e.g., "Execution/Production" or "QA/Review/Controls"), \
craft questions that probe that area using relatable language and data.

═══ ACKNOWLEDGMENTS — MICRO-INSIGHTS, NOT EMPATHY ═══

After each answer, your acknowledgment must contain a USEFUL observation \
based on what they told you — never generic empathy.

BAD:  "That makes sense." / "Got it." / "Thanks for sharing."
GOOD: "That's actually one of the top 3 patterns I see — when hooks don't \
       stop the scroll, it usually means the opening words aren't hitting a \
       nerve the reader cares about right now."
GOOD: "Interesting — solo founders who do their own outreach typically spend \
       12-15 hrs/week on it. That's usually the first thing worth automating."

═══ ADAPTING TO BUSINESS PROFILE ═══

If you have the user's business profile (team size, revenue, stage), use it:
- Solo founder: simpler frameworks, time-saving focus, low-cost benchmarks
- Growing team: process gaps, delegation bottlenecks, mid-market benchmarks
- Established: optimization metrics, competitive benchmarks, system-level gaps
Reference their scale naturally — "At your stage…" / "For a team of your size…"

═══ TONE & STYLE ═══

- Smart, caring advisor at a coffee shop — warm but incisive.
- Use "I" and "you" — it's a conversation, not a form.
- Include brief analogies or relatable comparisons when helpful.
- Show genuine curiosity about their specific situation.

═══ RESPONSE RULES ═══

1. ONE question per response. No exceptions.
2. 3-6 answer options per question. Always include "Something else" as the last.
3. You must ask a MINIMUM of 4 diagnostic questions. Aim for 5. \
   Do NOT signal "complete" before asking at least 4 questions. \
   After 6 questions you MUST signal completion. Absolute max is 7. \
   The user's earlier Q1 (outcome), Q2 (domain), Q3 (task) do NOT count — \
   your count starts from YOUR first diagnostic question.
4. Every option must be a specific, recognizable scenario (not generic labels).
5. The question text must be 1-2 sentences MAX — short, direct, punchy.
6. The insight field is MANDATORY. Max 10-12 words. One crisp stat or benchmark. \
   Never write full sentences. Never leave it generic or empty.
7. Acknowledgments: 1 sentence only. Contain a useful observation.

═══ RESPONSE FORMAT ═══

Respond in valid JSON only:

When asking a question:
{
  "status": "question",
  "acknowledgment": "1 sentence with a data-backed observation about their previous answer (skip for first question)",
  "insight": "Max 10-12 words — one crisp stat, benchmark, or pattern. No full sentences.",
  "question": "1-2 sentence max — the diagnostic question that follows from the insight.",
  "options": ["Specific scenario A", "Specific scenario B", "Specific scenario C", "Something else"],
  "section": "problems|rca_bridge|opportunities|deepdive",
  "section_label": "Crisp, specific label (e.g., 'Lead Response Speed')"
}

When diagnostic is complete:
{
  "status": "complete",
  "acknowledgment": "1 sentence power insight — their 'aha' moment",
  "summary": "2-3 sentence summary: what's broken, why, and the root cause. Include one final teaching insight."
}
"""


def _build_user_context(
    outcome: str,
    outcome_label: str,
    domain: str,
    task: str,
    diagnostic_context: dict[str, Any],
    rca_history: list[dict[str, str]],
    business_profile: dict[str, str] | None = None,
    crawl_summary: dict[str, Any] | None = None,
) -> str:
    """Build the rich user-context message sent alongside the system prompt."""
    parts = [
        "═══ USER PROFILE ═══",
        f"Outcome they want (Q1): {outcome_label}",
        f"Business domain  (Q2): {domain}",
        f"Specific task    (Q3): {task}",
    ]

    # ── Business scale profile (from scale questions) ──────────
    if business_profile:
        parts.append("\n═══ BUSINESS PROFILE (calibrate question depth to this) ═══")
        label_map = {
            "team_size": "Team Size",
            "current_stack": "Current Tech Stack",
            "business_stage": "Business Stage",
            "primary_channel": "Primary Acquisition Channel",
            "biggest_constraint": "Biggest Constraint",
        }
        for key, value in business_profile.items():
            label = label_map.get(key, key.replace("_", " ").title())
            parts.append(f"  • {label}: {value}")
        parts.append(
            "\n→ Use this profile to calibrate your questions. "
            "A solo pre-revenue founder needs different questions than a 50-person team doing ₹50L/mo. "
            "Reference their scale naturally — e.g., 'Since you're a solo founder…' or "
            "'With a team of 20+, the bottleneck is usually…'"
        )

    # ── Crawl / website analysis summary ───────────────────────
    if crawl_summary and crawl_summary.get("points"):
        parts.append("\n═══ WEBSITE ANALYSIS (from crawling their actual business site) ═══")
        parts.append("IMPORTANT: Reference these findings in your first 1-2 questions.")
        parts.append("This is REAL data about THEIR business — use it to make questions personal.")
        for pt in crawl_summary["points"]:
            parts.append(f"  • {pt}")
        parts.append(
            "\n→ Your first question should connect to something from their website. "
            "E.g., 'I see you're doing X on your site — [stat about X] — how is that working?'"
        )

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
                parts.append(f"\n═══ REAL-WORLD PROBLEM PATTERNS (mine these for insights to teach the user) ═══")
                parts.append("Each pattern below is a teaching opportunity. Reference specific ones in your insight field.")
                for i, item in enumerate(items, 1):
                    parts.append(f"  P{i}. {item}")

            elif key == "rca_bridge":
                parts.append(f"\n═══ DIAGNOSTIC SIGNALS — YOUR INSIGHT GOLDMINE (symptom → metric → root cause area) ═══")
                parts.append("CRITICAL: Use these metrics and KPIs in your 'insight' field. Quote specific numbers.")
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
                parts.append(f"\n═══ GROWTH OPPORTUNITIES (teach the user what 'good' looks like) ═══")
                parts.append("Reference these as benchmarks: 'Top performers do X…'")
                for i, item in enumerate(items, 1):
                    parts.append(f"  O{i}. {item}")

        # ── Strategies & frameworks ────────────────────────────
        strategies = diagnostic_context.get("strategies", "")
        if strategies:
            parts.append(f"\n═══ PROVEN STRATEGIES & FRAMEWORKS (name-drop these in insights) ═══")
            parts.append("When you reference a framework by name, users feel they're learning from an expert.")
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
                "Generate the NEXT question. Build on their answers. Drill deeper. "
                "REMEMBER: The 'insight' field is MANDATORY — teach them something from the knowledge base above."
            )
        else:
            parts.append(
                "\n→ Generate the NEXT question. Build on their answers. "
                "Drill deeper into the root cause. If you have enough info "
                "you may signal 'complete', or ask one more to sharpen the diagnosis. "
                "The 'insight' field is still MANDATORY for any question you ask."
            )
    else:
        parts.append(
            "\n→ This is the FIRST question. Start with a compelling insight — "
            "a stat or pattern from the knowledge base that immediately "
            "makes the user think 'huh, I didn't know that.' Then ask "
            "your diagnostic question. The 'insight' field MUST contain "
            "a specific, educational teaching moment."
        )

    return "\n".join(parts)


async def generate_next_rca_question(
    outcome: str,
    outcome_label: str,
    domain: str,
    task: str,
    diagnostic_context: dict[str, Any],
    rca_history: list[dict[str, str]],
    business_profile: dict[str, str] | None = None,
    crawl_summary: dict[str, Any] | None = None,
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
        outcome, outcome_label, domain, task, diagnostic_context, rca_history,
        business_profile=business_profile,
        crawl_summary=crawl_summary,
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
