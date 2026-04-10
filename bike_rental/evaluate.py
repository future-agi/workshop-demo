"""
Bike Rental Agent - Evaluation.

Runs simulation and evaluates results using FutureAGI's built-in templates.
Focuses on tool calling accuracy, conversation coherence, and hallucination detection.

Eval templates used:
- task_completion: Did the agent complete the booking/info request?
- evaluate_function_calling: Were the right tools called correctly?
- is_helpful: Was the agent proactive?
- conversation_coherence: Did the multi-turn flow make sense?
- detect_hallucination: Did the agent make up info instead of using tools?
- is_concise: Were responses appropriately brief?

Run: python bike_rental/evaluate.py
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fi.simulate import evaluate_report
from bike_rental.agent_with_simulate import main as run_simulation


EVAL_TEMPLATES = [
    "task_completion",
    "evaluate_function_calling",
    "is_helpful",
    "conversation_coherence",
    "detect_hallucination",
    "is_concise",
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
        print(f"\n--- {name} ({result.persona.persona['role']}) ---")

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
    print("Expected issues:")
    print("- evaluate_function_calling: May show incorrect tool usage patterns")
    print("- detect_hallucination: Agent may guess instead of using tools")
    print("- task_completion: Missing confirmation before booking")
    print("Run optimize.py to improve the prompt with ProTeGi.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
