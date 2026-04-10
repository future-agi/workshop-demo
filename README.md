# AI Workshop Demo: Agents + FutureAGI

Three demo agents showcasing how to build, simulate, evaluate, and optimize AI agents using [FutureAGI](https://futureagi.com).

Each agent has two states:
- **State A** (basic): Standalone agent with an intentionally "good but not great" prompt
- **State B** (with FutureAGI): Same agent + simulation, evaluation, and prompt optimization

## Agents

| Agent | Type | Framework | Optimizer |
|-------|------|-----------|-----------|
| `simple_chat/` | Chat assistant | Groq/OpenAI SDK | MetaPrompt |
| `bike_rental/` | Bike rental with tool calls | LangChain | ProTeGi |
| `hotel_voice/` | Voice hotel receptionist | LiveKit | GEPA |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run any agent
python simple_chat/agent.py          # Simple chat
python bike_rental/agent.py          # Bike rental
python hotel_voice/agent.py dev      # Voice agent

# 4. Run simulation + evaluation
python simple_chat/agent_with_simulate.py
python simple_chat/evaluate.py

# 5. Run prompt optimization
python simple_chat/optimize.py
```

## LLM Configuration

Agents are LLM-agnostic. Set your provider in `.env`:

```bash
LLM_PROVIDER=groq                     # groq | openai | anthropic
MODEL_NAME=llama-3.3-70b-versatile    # any model for your provider
```

## Required API Keys

| Key | Required For |
|-----|-------------|
| `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | All agents (pick one) |
| `FI_API_KEY` + `FI_SECRET_KEY` | Simulation, evaluation, optimization |
| `LIVEKIT_URL` + `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET` | hotel_voice agent only |
| `DEEPGRAM_API_KEY` | hotel_voice agent only |

## Project Structure

```
workshop-demo/
├── config.py              # Shared LLM config (provider-agnostic)
├── simple_chat/           # Agent 1: Simple chat assistant
├── bike_rental/           # Agent 2: LangChain bike rental with tools
├── hotel_voice/           # Agent 3: LiveKit voice hotel agent
└── docs/                  # Guides for extending this project
```

## Documentation

- [Getting Started](docs/getting-started.md) - Setup and installation
- [Adding New Agents](docs/adding-new-agents.md) - Template for creating more agents
- [FutureAGI Integration](docs/futureagi-integration.md) - How simulate, evaluate, and optimize work
