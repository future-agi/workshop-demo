"""
Hotel Voice Agent - Prompt Optimization.

Uses GEPAOptimizer (Genetic-Pareto Evolutionary) to improve the voice agent's
system prompt. GEPA is the most powerful optimizer, ideal for complex agents
with competing objectives (conciseness vs warmth vs accuracy vs voice-friendliness).

Optimizer: GEPAOptimizer
- Evaluates current prompts against dataset
- Reflection LLM analyzes failures
- Evolutionary mutations generate improved variants
- Pareto-aware selection balances multiple objectives
- Best for: complex, multi-objective optimization

Run: python hotel_voice/optimize.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_litellm_model
from fi.opt.generators import LiteLLMGenerator
from fi.opt.optimizers import GEPAOptimizer
from fi.opt.base.evaluator import Evaluator
from fi.opt.datamappers import BasicDataMapper
from fi.evals.metrics import CustomLLMJudge
from fi.evals.llm import LiteLLMProvider

# The deliberately suboptimal voice prompt
INITIAL_PROMPT = (
    "You are a hotel receptionist at The Grand Hotel. Help guests with:\n"
    "- Room availability and pricing\n"
    "- Making reservations\n"
    "- Cancellations and modifications\n"
    "- Hotel amenities information\n\n"
    "Use the available tools to look up information. Be professional and helpful.\n"
    "Always provide complete details including:\n"
    "* Room type options\n"
    "* Pricing breakdowns\n"
    "* Available dates\n"
    "* Amenity lists\n\n"
    "Guest says: {guest_message}"
)

# Dataset: voice conversation scenarios with ideal spoken responses
DATASET = [
    {
        "guest_message": "Hi, I need a suite for next Thursday through Sunday.",
        "answer": "I'd be happy to check that for you! A suite is three-fifty a night. Let me look up availability for Thursday to Sunday. Could I get your name for the reservation?",
    },
    {
        "guest_message": "What's your cheapest room?",
        "answer": "Our Standard Room is a hundred twenty a night. It's a comfortable room with a queen bed and city view. Would you like to book one?",
    },
    {
        "guest_message": "We're planning our anniversary. What do you recommend?",
        "answer": "Congratulations! For a special occasion, I'd recommend our Presidential Suite. It has two bedrooms, a dining area, and butler service. It's six hundred a night. We also have a wonderful spa and rooftop lounge. Shall I check availability?",
    },
    {
        "guest_message": "I need 5 rooms for a conference next month.",
        "answer": "I can help with that. What type of rooms are you looking for, and what dates? We have Standard rooms at a hundred twenty, Deluxe at two hundred, and Suites at three-fifty per night.",
    },
    {
        "guest_message": "Do you have a pool?",
        "answer": "Yes, we have an outdoor pool open from seven A.M. to ten P.M. We also have a fitness center, spa, and restaurant. Is there anything specific you'd like to know?",
    },
    {
        "guest_message": "Can I cancel reservation RES-ABC123?",
        "answer": "Let me look that up for you. Could you confirm the name on the reservation? I want to make sure I'm cancelling the right one.",
    },
    {
        "guest_message": "What amenities come with the deluxe room?",
        "answer": "The Deluxe Room includes Wi-Fi, TV, a mini bar, coffee maker, bathrobes, and room service. It's two hundred a night with a king bed and panoramic view. Would you like to book one?",
    },
    {
        "guest_message": "My name is Priya, P-R-I-Y-A. I'd like a standard room for Friday.",
        "answer": "Got it, Priya. Let me check Standard Room availability for Friday. Just one night, or will you be staying longer?",
    },
]


def main():
    model = get_litellm_model()
    print("Hotel Voice Agent - Prompt Optimization")
    print(f"Model: {model}")
    print(f"Optimizer: GEPAOptimizer (evolutionary)")
    print(f"Dataset size: {len(DATASET)} examples")
    print("=" * 60)

    # Step 1: LLM-as-a-judge for voice-specific evaluation
    provider = LiteLLMProvider()
    judge = CustomLLMJudge(
        provider=provider,
        config={
            "name": "voice_agent_judge",
            "grading_criteria": (
                "You are evaluating a VOICE hotel receptionist's response.\n"
                "This is SPOKEN, not written. Score 0.0 to 1.0 based on:\n"
                "- Voice-friendliness: No bullet points, markdown, or text formatting. "
                "Numbers spoken naturally (e.g., 'three-fifty' not '$350'). (0.3 weight)\n"
                "- Conciseness: Short, speakable sentences. No info dumps. (0.25 weight)\n"
                "- Warmth: Friendly, welcoming hotel receptionist tone. (0.2 weight)\n"
                "- Task completion: Does it move toward resolving the guest's request? (0.25 weight)\n"
                "The 'response' is the agent's spoken reply. The 'expected_response' is the ideal.\n"
                "Score 1.0 for natural, concise, warm, task-completing responses."
            ),
        },
        model=model,
        temperature=0.3,
    )
    evaluator = Evaluator(metric=judge)

    # Step 2: Data mapper
    data_mapper = BasicDataMapper(
        key_map={
            "response": "generated_output",
            "expected_response": "answer",
        }
    )

    # Step 3: GEPA optimizer
    optimizer = GEPAOptimizer(
        reflection_model=model,
        generator_model=model,
    )

    # Step 4: Show the starting prompt
    print("\nSTARTING PROMPT:")
    print("-" * 40)
    print(INITIAL_PROMPT)
    print("-" * 40)

    # Step 5: Run optimization
    print("\nRunning GEPA optimization (this may take a few minutes)...\n")
    result = optimizer.optimize(
        evaluator=evaluator,
        data_mapper=data_mapper,
        dataset=DATASET,
        initial_prompts=[INITIAL_PROMPT],
        max_metric_calls=200,
    )

    # Step 6: Show results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nFinal Score: {result.final_score:.4f}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Total Evaluations: {result.total_evaluations}")
    print(f"Early Stopped: {result.early_stopped}")

    if result.history:
        print("\nScore progression:")
        for i, h in enumerate(result.history):
            print(f"  Generation {i + 1}: {h.average_score:.4f}")

    print("\nOPTIMIZED PROMPT:")
    print("-" * 40)
    print(result.best_generator.get_prompt_template())
    print("-" * 40)

    print("\nKey improvements expected:")
    print("- Removed bullet points and markdown (voice-inappropriate)")
    print("- Added conciseness guidance (short sentences for speech)")
    print("- Added warmth and personality")
    print("- Added clarification behavior (spell-back names/dates)")
    print("- Natural number formatting ('three-fifty' not '$350')")


if __name__ == "__main__":
    main()
