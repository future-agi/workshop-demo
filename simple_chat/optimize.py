"""
Simple Chat Agent - Prompt Optimization.

Uses MetaPromptOptimizer to improve the agent's system prompt.
Demonstrates how optimization identifies prompt weaknesses and rewrites
the prompt to be more helpful, concise, and personality-driven.

Optimizer: MetaPromptOptimizer
- Evaluates current prompt on dataset
- Teacher LLM analyzes failures and hypothesizes improvements
- Rewrites entire prompt each iteration
- Best for: simple, single-prompt agents

Run: python simple_chat/optimize.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_litellm_model
from fi.opt.generators import LiteLLMGenerator
from fi.opt.optimizers import MetaPromptOptimizer
from fi.opt.base.evaluator import Evaluator
from fi.opt.datamappers import BasicDataMapper
from fi.evals.metrics import CustomLLMJudge
from fi.evals.llm import LiteLLMProvider

# The deliberately suboptimal prompt we want to improve
INITIAL_PROMPT = (
    "You are an AI assistant. Answer user questions. Provide information when asked.\n\n"
    "User question: {question}"
)

# Dataset: diverse QA pairs that expose the prompt's weaknesses
DATASET = [
    {
        "question": "What is quantum entanglement? I'm studying for my physics exam.",
        "answer": "Quantum entanglement is when two particles become linked so that measuring one instantly affects the other, regardless of distance. For your exam, remember: it's about correlated quantum states, not faster-than-light communication.",
    },
    {
        "question": "My laptop keeps crashing during meetings. Help!",
        "answer": "That's frustrating! Let's fix this step by step: 1) Check if you're running low on RAM during meetings, 2) Update your video call app, 3) Check your laptop's temperature. Which one should we start with?",
    },
    {
        "question": "How do I send photos on my new smartphone?",
        "answer": "Open your messaging app, tap the camera or paperclip icon at the bottom, select the photo you want to send, then tap the send button. Would you like me to walk you through it for a specific app?",
    },
    {
        "question": "Quick - pros and cons of microservices vs monolith? Meeting in 10 min.",
        "answer": "Microservices: independent scaling, team autonomy, tech flexibility. Downsides: complex ops, network overhead. Monolith: simpler deployment, easier debugging. Downsides: scaling bottlenecks, tight coupling. For most startups, start monolith, migrate later.",
    },
    {
        "question": "I'm feeling overwhelmed with work. Any advice?",
        "answer": "I hear you. Try this: write down your top 3 priorities for today, tackle the hardest one first when your energy is highest, and give yourself permission to say no to non-essential tasks. Small steps add up.",
    },
    {
        "question": "Explain the difference between machine learning and deep learning.",
        "answer": "Machine learning is the broad field where computers learn from data. Deep learning is a subset that uses neural networks with many layers. Think of ML as the category and DL as one powerful technique within it, especially good at images, text, and speech.",
    },
]


def main():
    model = get_litellm_model()
    print("Simple Chat Agent - Prompt Optimization")
    print(f"Model: {model}")
    print(f"Optimizer: MetaPromptOptimizer")
    print(f"Dataset size: {len(DATASET)} examples")
    print("=" * 60)

    # Step 1: Set up the teacher generator (powers the optimizer)
    teacher = LiteLLMGenerator(
        model=model,
        prompt_template="{prompt}",
    )

    # Step 2: Set up evaluation - LLM-as-a-judge scoring helpfulness + conciseness
    provider = LiteLLMProvider()
    judge = CustomLLMJudge(
        provider=provider,
        config={
            "name": "chat_quality_judge",
            "model": model,
            "grading_criteria": (
                "You are evaluating an AI assistant's response to a user question.\n"
                "Score from 0.0 to 1.0 based on these criteria:\n"
                "- Helpfulness: Does it actually answer the question? (0.3 weight)\n"
                "- Conciseness: Is it brief and to the point, not rambling? (0.3 weight)\n"
                "- Tone: Is it warm, natural, and not robotic? (0.2 weight)\n"
                "- Actionability: Does it give practical next steps? (0.2 weight)\n"
                "The 'response' field is the AI's answer. The 'expected_response' is the ideal answer.\n"
                "Score 1.0 if the response matches or exceeds the ideal. Score 0.0 if unhelpful or poor."
            ),
        },
    )
    evaluator = Evaluator(metric=judge)

    # Step 3: Map generated output to evaluator inputs
    data_mapper = BasicDataMapper(
        key_map={
            "response": "generated_output",
            "expected_response": "answer",
        }
    )

    # Step 4: Create optimizer
    optimizer = MetaPromptOptimizer(teacher_generator=teacher)

    # Step 5: Show the starting prompt
    print("\nSTARTING PROMPT:")
    print("-" * 40)
    print(INITIAL_PROMPT)
    print("-" * 40)

    # Step 6: Run optimization
    print("\nRunning optimization (5 rounds)...\n")
    result = optimizer.optimize(
        evaluator=evaluator,
        data_mapper=data_mapper,
        dataset=DATASET,
        initial_prompts=[INITIAL_PROMPT],
        task_description=(
            "Optimize this chat assistant's system prompt to make it more helpful, "
            "concise, warm in tone, and actionable. The assistant should have a clear "
            "personality, avoid unnecessary verbosity, and provide practical guidance."
        ),
        num_rounds=5,
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
