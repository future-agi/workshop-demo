# Bike Rental Agent

A LangChain agent with tool calling for a bike rental business. Demonstrates FutureAGI simulation, evaluation (including function calling analysis), and optimization with ProTeGi.

## Files

| File | Description |
|------|-------------|
| `tools.py` | Mock database + 6 LangChain tools (list, price, availability, details, book, cancel) |
| `agent.py` | **State A** - LangChain agent with tools |
| `agent_with_simulate.py` | **State B** - Wrapped with LangChainAgentWrapper |
| `evaluate.py` | Evaluate with function calling + hallucination detection |
| `optimize.py` | Optimize with ProTeGi (beam search + textual gradients) |
| `notebook.ipynb` | Jupyter walkthrough |

## Run

```bash
# State A: Interactive chat
python bike_rental/agent.py

# State B: Simulation
python bike_rental/agent_with_simulate.py

# Evaluation
python bike_rental/evaluate.py

# Optimization
python bike_rental/optimize.py
```

## Inventory

| Vehicle | ID | Hourly | Daily | Status |
|---------|----|--------|-------|--------|
| Mountain Bike Pro | MTB-001 | $8 | $45 | Available |
| City Cruiser | CTY-002 | $6 | $35 | Available |
| Electric Scooter X | ESC-003 | $10 | $55 | Available |
| Road Bike Elite | RDB-004 | $12 | $65 | Available |
| E-Bike Comfort | EBK-005 | $15 | $80 | Unavailable |
| Tandem Fun Ride | TND-006 | $18 | $95 | Available |

## Deliberate Prompt Weaknesses (State A)

- No guidance on when to use tools vs guessing
- No confirmation before booking
- No price formatting
- No handling of unavailable vehicles
- No upsell/recommendation behavior

## Evaluation Templates

- `task_completion` - Did it complete the request?
- `evaluate_function_calling` - Were tools called correctly?
- `is_helpful` - Was it proactive?
- `conversation_coherence` - Did multi-turn flow make sense?
- `detect_hallucination` - Did it make up info?
- `is_concise` - Were responses brief?

## Optimizer

**ProTeGi** - Beam search with textual gradients. Identifies specific failure patterns (wrong tool calls, missing confirmations) and generates targeted fixes. Best for agents with debuggable failure modes.
