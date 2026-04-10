"""
Simple Chat Agent - Evaluation.

Runs the simulation and evaluates the results using FutureAGI's built-in
evaluation templates. Demonstrates how evaluation catches prompt weaknesses.

Eval templates used:
- task_completion: Did it answer the question?
- is_helpful: Was the response useful?
- is_concise: Was it brief enough?
- tone: Was the tone appropriate?
- no_llm_reference: Did it avoid "As an AI..." phrasing?

Run: python simple_chat/evaluate.py
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fi.simulate import evaluate_report
from simple_chat.agent_with_simulate import main as run_simulation


EVAL_TEMPLATES = [
    "task_completion",
    "is_helpful",
    "is_concise",
    "tone",
    "no_llm_reference",
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

    # Step 3: Print evaluation results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for result in report.results:
        name = result.persona.persona["name"]
        print(f"\n--- {name} ---")

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
    print("Note: Low scores on is_concise and tone are expected with the")
    print("current prompt. Run optimize.py to see how optimization improves it.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
