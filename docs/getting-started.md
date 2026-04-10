# Getting Started

## Prerequisites

- Python 3.10+
- API keys (see below)

## Installation

```bash
# Clone or navigate to the project
cd workshop-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

### Required for all agents

| Variable | Where to get it |
|----------|----------------|
| `LLM_PROVIDER` | `groq`, `openai`, or `anthropic` |
| `MODEL_NAME` | Model name for your provider |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |

### Required for simulation, evaluation, and optimization

| Variable | Where to get it |
|----------|----------------|
| `FI_API_KEY` | [app.futureagi.com](https://app.futureagi.com) |
| `FI_SECRET_KEY` | [app.futureagi.com](https://app.futureagi.com) |

### Required for voice agent only

| Variable | Where to get it |
|----------|----------------|
| `LIVEKIT_URL` | [cloud.livekit.io](https://cloud.livekit.io) or self-hosted |
| `LIVEKIT_API_KEY` | LiveKit dashboard |
| `LIVEKIT_API_SECRET` | LiveKit dashboard |
| `DEEPGRAM_API_KEY` | [console.deepgram.com](https://console.deepgram.com) |

## Quick Start

### Agent 1: Simple Chat

```bash
# Basic chat
python simple_chat/agent.py

# Simulate + evaluate
python simple_chat/agent_with_simulate.py
python simple_chat/evaluate.py

# Optimize the prompt
python simple_chat/optimize.py
```

### Agent 2: Bike Rental

```bash
# Interactive chat with tool calls
python bike_rental/agent.py

# Simulate + evaluate
python bike_rental/agent_with_simulate.py
python bike_rental/evaluate.py

# Optimize with ProTeGi
python bike_rental/optimize.py
```

### Agent 3: Hotel Voice

```bash
# Start voice agent (keep running)
python hotel_voice/agent.py dev

# In another terminal: simulate
python hotel_voice/agent_with_simulate.py
python hotel_voice/evaluate.py

# Optimize (standalone, no voice agent needed)
python hotel_voice/optimize.py
```

### Jupyter Notebooks

```bash
jupyter notebook
# Open any agent's notebook.ipynb
```

## Switching LLM Providers

Change two lines in `.env`:

```bash
# Use Groq (fast, free tier)
LLM_PROVIDER=groq
MODEL_NAME=llama-3.3-70b-versatile

# Use OpenAI
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o

# Use Anthropic
LLM_PROVIDER=anthropic
MODEL_NAME=claude-sonnet-4-20250514
```

All agents will automatically use the new provider. No code changes needed.
