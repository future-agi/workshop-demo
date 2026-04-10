# Hotel Voice Agent

A LiveKit voice agent for hotel booking with STT/LLM/TTS pipeline and tool calling. Demonstrates FutureAGI simulation with audio recording, voice-specific evaluation, and GEPA evolutionary optimization.

## Files

| File | Description |
|------|-------------|
| `tools.py` | Mock hotel database + tool functions (rooms, prices, booking, etc.) |
| `agent.py` | **State A** - LiveKit voice agent with Deepgram STT + LLM + OpenAI TTS |
| `agent_with_simulate.py` | **State B** - Simulate SDK with LiveKit engine + audio recording |
| `evaluate.py` | Evaluate with voice-specific + customer agent templates |
| `optimize.py` | Optimize with GEPAOptimizer (evolutionary, multi-objective) |
| `notebook.ipynb` | Jupyter walkthrough |

## Run

```bash
# State A: Voice agent (requires LiveKit, Deepgram, OpenAI TTS)
python hotel_voice/agent.py dev

# State B: Simulation (requires voice agent running)
python hotel_voice/agent_with_simulate.py

# Evaluation
python hotel_voice/evaluate.py

# Optimization (standalone, no voice agent needed)
python hotel_voice/optimize.py
```

## Room Types

| Room | Price/Night | Capacity |
|------|------------|----------|
| Standard | $120 | 2 guests |
| Deluxe | $200 | 2 guests |
| Suite | $350 | 3 guests |
| Presidential | $600 | 4 guests |

## Deliberate Prompt Weaknesses (State A)

- Uses bullet points and markdown formatting (bad for spoken output)
- Encourages verbose "complete details" dumps
- No conciseness guidance for voice
- No clarification behavior for names/dates
- No interruption handling

## Evaluation Templates

- `task_completion` - Did it handle the booking request?
- `is_concise` - Critical for voice: short, speakable responses
- `tone` - Warm, professional hotel receptionist
- `conversation_quality` - Overall voice conversation quality
- `context_retention` - Remembered details across conversation
- `clarification_seeking` - Asked for clarification when needed
- `loop_detection` - Avoided repeating itself

## Optimizer

**GEPAOptimizer** - Genetic-Pareto evolutionary optimization. Uses a reflection LLM to analyze failures and guides evolutionary mutations. Pareto-aware selection balances competing objectives (conciseness + warmth + accuracy + voice-friendliness). The most powerful optimizer, best for complex multi-objective tasks.

## Voice Pipeline

```
Caller Speech -> Deepgram STT -> LLM (Groq/OpenAI) -> OpenAI TTS -> Spoken Response
                                  |
                           Tool Calls (rooms, booking, amenities)
```
