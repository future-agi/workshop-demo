# Simple Chat Agent

A basic LLM chat loop demonstrating FutureAGI simulation, evaluation, and optimization.

## Files

| File | Description |
|------|-------------|
| `agent.py` | **State A** - Basic chat agent (LLM in a loop) |
| `agent_with_simulate.py` | **State B** - Same agent wrapped with Simulate SDK |
| `evaluate.py` | Evaluate simulation results with FutureAGI templates |
| `optimize.py` | Optimize the system prompt with MetaPromptOptimizer |
| `notebook.ipynb` | Jupyter walkthrough of all steps |

## Run

```bash
# State A: Basic chat
python simple_chat/agent.py

# State B: Simulation
python simple_chat/agent_with_simulate.py

# Evaluation
python simple_chat/evaluate.py

# Optimization
python simple_chat/optimize.py
```

## Deliberate Prompt Weaknesses (State A)

The system prompt is intentionally suboptimal:
- Vague identity ("You are an AI assistant")
- No conciseness guidance
- No tone or personality direction
- No follow-up or guidance behavior

## Evaluation Templates

- `task_completion` - Did it answer the question?
- `is_helpful` - Was the response useful?
- `is_concise` - Was it brief enough?
- `tone` - Was the tone appropriate?
- `no_llm_reference` - Did it avoid "As an AI..." phrasing?

## Optimizer

**MetaPromptOptimizer** - Evaluates prompt, analyzes failures, rewrites entirely each round. Best for simple single-prompt agents.
