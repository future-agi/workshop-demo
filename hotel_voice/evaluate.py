"""
Hotel Voice Agent - Evaluation.

Runs simulation and evaluates using voice-specific and customer agent templates.

Eval templates used:
- task_completion: Did the agent handle the booking request?
- is_concise: Critical for voice - short, speakable responses
- tone: Warm, professional hotel receptionist tone
- conversation_quality: Overall voice conversation quality (Customer Agent)
- context_retention: Did it remember details across the conversation?
- clarification_seeking: Did it ask for clarification when needed?
- loop_detection: Did it avoid repeating itself?

Run: python hotel_voice/evaluate.py
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fi.simulate import evaluate_report
from hotel_voice.agent_with_simulate import main as run_simulation


EVAL_TEMPLATES = [
    "task_completion",
    "is_concise",
    "tone",
    "conversation_quality",
    "context_retention",
    "clarification_seeking",
    "loop_detection",
]


async def main():
    # Step 1: Run simulation
    print("Step 1: Running simulation...\n")
    report = await run_simulation()

    # Step 2: Evaluate
    print("\nStep 2: Evaluating transcripts...\n")
    report = evaluate_report(
        report,
        eval_templates=EVAL_TEMPLATES,
        model_name="turing_flash",
    )

    # Step 3: Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for result in report.results:
        name = result.persona.persona["name"]
        role = result.persona.persona["role"]
        print(f"\n--- {name} ({role}) ---")

        if result.evaluation:
            for template, scores in result.evaluation.items():
                if "error" in scores:
                    print(f"  {template}: ERROR - {scores['error']}")
                else:
                    score = scores.get("score", "N/A")
                    output = scores.get("output", "N/A")
                    reason = scores.get("reason", "")
                    status = "PASS" if isinstance(score, (int, float)) and score >= 0.7 else "FAIL"
                    print(f"  {template}: {output} ({status})")
                    if reason:
                        print(f"    Reason: {reason[:200]}")
        else:
            print("  No evaluation data.")

    print("\n" + "=" * 60)
    print("Expected issues with current voice prompt:")
    print("- is_concise: FAIL - prompt encourages verbose 'complete details' dumps")
    print("- conversation_quality: Lower than ideal - markdown formatting in voice")
    print("- clarification_seeking: FAIL - doesn't ask to spell back names/dates")
    print("Run optimize.py to improve with GEPA evolutionary optimization.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
