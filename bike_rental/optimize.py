"""
Bike Rental Agent - Prompt Optimization.

Uses ProTeGi optimizer to improve the agent's system prompt.
ProTeGi uses beam search with targeted critiques - ideal for agents
with specific failure modes like wrong tool calls or missing confirmations.

Optimizer: ProTeGi
- Identifies failure patterns in current prompt
- Generates targeted critiques ("textual gradients")
- Creates improved prompt variations for each critique
- Keeps top N candidates via beam search
- Best for: agents with specific, debuggable failure modes

Run: python bike_rental/optimize.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_litellm_model
from fi.opt.generators import LiteLLMGenerator
from fi.opt.optimizers import ProTeGi
from fi.opt.base.evaluator import Evaluator
from fi.opt.datamappers import BasicDataMapper
from fi.evals.metrics import CustomLLMJudge
from fi.evals.llm import LiteLLMProvider

# The deliberately suboptimal prompt
INITIAL_PROMPT = (
    "You are a bike rental assistant. Help customers with bike rentals. "
    "You have access to tools for checking prices, availability, and making bookings.\n\n"
    "Customer message: {customer_message}"
)

# Dataset: bike rental scenarios with expected ideal responses
DATASET = [
    {
        "customer_message": "What's the cheapest bike you have for 3 hours?",
        "answer": "The cheapest option is the City Cruiser at $6/hour. For 3 hours that would be $18. Would you like me to check availability and book it for you?",
    },
    {
        "customer_message": "I need 4 bikes for a family day trip. Kids are 8 and 12.",
        "answer": "For a family day trip I'd recommend City Cruisers - they're comfortable and easy to ride. At $35/day each, that's $140 total for 4 bikes. Should I check availability for your date?",
    },
    {
        "customer_message": "Book me the road bike for tomorrow, name is Mika.",
        "answer": "Before I book, let me confirm: Road Bike Elite (RDB-004) for Mika, tomorrow. How many hours would you like? The rate is $12/hour or $65 for a full day.",
    },
    {
        "customer_message": "I want the e-bike for Saturday.",
        "answer": "I'm sorry, the E-Bike Comfort is currently unavailable for maintenance. I'd recommend the City Cruiser ($6/hr) for a relaxed ride, or the Mountain Bike Pro ($8/hr) for something more versatile. Would either work?",
    },
    {
        "customer_message": "I've never rented a bike before. What should I get to ride around the park?",
        "answer": "For a casual park ride, I'd recommend the City Cruiser - it's our most comfortable option at $6/hour with a basket and bell. It's beginner-friendly. How long were you thinking of riding?",
    },
    {
        "customer_message": "What are the specs on the mountain bike?",
        "answer": "Mountain Bike Pro: aluminum frame, 26-inch wheels, 21 gears, 13 kg. It has front suspension, great for trails and rougher terrain. Priced at $8/hr or $45/day.",
    },
    {
        "customer_message": "Cancel booking BK-ABC123 please.",
        "answer": "I'll cancel booking BK-ABC123 right away. Your refund will be processed. Is there anything else I can help you with?",
    },
    {
        "customer_message": "Which bike is best for a long distance ride, maybe 50km?",
        "answer": "For a 50km ride, the Road Bike Elite is your best bet - it's a carbon frame road bike at just 8 kg with 22 Shimano gears. At $12/hr or $65/day, it's built for performance. Want me to check availability?",
    },
]


def main():
    model = get_litellm_model()
    print("Bike Rental Agent - Prompt Optimization")
    print(f"Model: {model}")
    print(f"Optimizer: ProTeGi (beam search + textual gradients)")
    print(f"Dataset size: {len(DATASET)} examples")
    print("=" * 60)

    # Step 1: Set up the teacher generator
    teacher = LiteLLMGenerator(
        model=model,
        prompt_template="{prompt}",
    )

    # Step 2: LLM-as-a-judge for tool-aware evaluation
    provider = LiteLLMProvider()
    judge = CustomLLMJudge(
        provider=provider,
        config={
            "name": "bike_rental_judge",
            "model": model,
            "grading_criteria": (
                "You are evaluating a bike rental assistant's response.\n"
                "Score from 0.0 to 1.0 based on:\n"
                "- Tool awareness: Does it mention using tools to check info instead of guessing? (0.25)\n"
                "- Confirmation: Does it confirm details before booking? (0.25)\n"
                "- Recommendation quality: Does it suggest the right bike for the situation? (0.25)\n"
                "- Helpfulness: Does it guide the customer toward a decision? (0.25)\n"
                "The 'response' is the agent's answer. The 'expected_response' is the ideal.\n"
                "Score 1.0 if the response matches the ideal's quality. Score 0.0 if unhelpful."
            ),
        },
    )
    evaluator = Evaluator(metric=judge)

    # Step 3: Data mapper
    data_mapper = BasicDataMapper(
        key_map={
            "response": "generated_output",
            "expected_response": "answer",
        }
    )

    # Step 4: Create ProTeGi optimizer
    optimizer = ProTeGi(
        teacher_generator=teacher,
        num_gradients=4,
        errors_per_gradient=4,
        prompts_per_gradient=1,
        beam_size=4,
    )

    # Step 5: Show the starting prompt
    print("\nSTARTING PROMPT:")
    print("-" * 40)
    print(INITIAL_PROMPT)
    print("-" * 40)

    # Step 6: Run optimization
    print("\nRunning optimization (3 rounds, beam size 4)...\n")
    result = optimizer.optimize(
        evaluator=evaluator,
        data_mapper=data_mapper,
        dataset=DATASET,
        initial_prompts=[INITIAL_PROMPT],
        num_rounds=3,
        eval_subset_size=len(DATASET),
    )

    # Step 7: Show results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nFinal Score: {result.final_score:.4f}")
    print(f"Iterations: {len(result.history)}")

    if result.history:
        print("\nScore progression:")
        for i, h in enumerate(result.history):
            print(f"  Round {i + 1}: {h.average_score:.4f}")

    optimized = result.best_generator.get_prompt_template()
    print("\nOPTIMIZED PROMPT:")
    print("-" * 40)
    print(optimized)
    print("-" * 40)

    print("\nCOMPARISON:")
    print(f"  Before: '{INITIAL_PROMPT[:100]}...'")
    print(f"  After:  '{optimized[:100]}...'")
    print(f"  Score: {result.final_score:.4f}")


if __name__ == "__main__":
    main()
