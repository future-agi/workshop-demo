# Adding New Agents

Follow this template to create new agents in the same format as the existing ones.

## Step 1: Create the folder structure

```
my_agent/
├── README.md
├── tools.py                  # If your agent uses tools
├── agent.py                  # State A: basic agent
├── agent_with_simulate.py    # State B: with Simulate SDK
├── evaluate.py               # Evaluation
├── optimize.py               # Prompt optimization
└── notebook.ipynb            # Jupyter walkthrough
```

## Step 2: Define tools (if applicable)

```python
# my_agent/tools.py
from langchain_core.tools import tool  # For LangChain agents
# or use standalone functions for LiveKit agents

MOCK_DATA = {
    # Your mock database
}

@tool
def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result for {param}"
```

## Step 3: Write the basic agent (State A)

```python
# my_agent/agent.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm_client, get_model_name  # or get_langchain_llm

# Deliberately suboptimal prompt (for evaluation to catch issues)
SYSTEM_PROMPT = "You are a [role]. Help users with [task]."

def chat():
    client = get_llm_client()
    # ... your chat loop
```

### Tips for prompt weaknesses

Make the prompt functional but flawed in specific, fixable ways:
- **Missing guidance**: Don't tell the agent when to use tools
- **No confirmation**: Let it act without confirming details
- **Wrong format**: Use text formatting in voice agents
- **Too verbose**: Don't set conciseness expectations
- **No personality**: Keep it generic and robotic

## Step 4: Wrap with Simulate SDK (State B)

### For simple/custom agents:

```python
from fi.simulate import AgentWrapper, AgentInput, AgentResponse

class MyAgentWrapper(AgentWrapper):
    async def call(self, input: AgentInput) -> AgentResponse:
        # Call your LLM with input.messages
        response = client.chat.completions.create(
            model=model, messages=input.messages
        )
        return AgentResponse(content=response.choices[0].message.content)
```

### For LangChain agents:

```python
from fi.simulate import LangChainAgentWrapper

wrapper = LangChainAgentWrapper(agent=executor, system_prompt=SYSTEM_PROMPT)
```

### For LiveKit voice agents:

```python
from fi.simulate import AgentDefinition, SimulatorAgentDefinition

agent_def = AgentDefinition(
    name="My Agent", url=LIVEKIT_URL, room_name="test-room",
    system_prompt=SYSTEM_PROMPT,
)
```

### Create test scenarios:

```python
from fi.simulate import Scenario, Persona

scenario = Scenario(
    name="my-test",
    dataset=[
        Persona(
            persona={"name": "Alice", "mood": "curious"},
            situation="She wants to know about X.",
            outcome="Clear, helpful answer about X.",
        ),
    ],
)
```

## Step 5: Add evaluation

Choose eval templates based on your agent type:

### Chat agents
- `task_completion`, `is_helpful`, `is_concise`, `tone`, `no_llm_reference`

### Tool-calling agents
- `task_completion`, `evaluate_function_calling`, `detect_hallucination`, `conversation_coherence`

### Voice agents
- `task_completion`, `is_concise`, `tone`, `conversation_quality`, `context_retention`, `clarification_seeking`

### Customer support agents
- `conversation_quality`, `context_retention`, `query_handling`, `objection_handling`, `human_escalation`

```python
from fi.simulate import evaluate_report

report = evaluate_report(
    report,
    eval_templates=["task_completion", "is_helpful", "is_concise"],
    model_name="turing_flash",
)
```

## Step 6: Add optimization

Choose an optimizer based on your agent type:

| Optimizer | Best For | Key Idea |
|-----------|----------|----------|
| **MetaPromptOptimizer** | Simple single-prompt agents | Evaluate, hypothesize failure, rewrite entire prompt |
| **ProTeGi** | Agents with specific failure modes | Beam search + targeted critiques (textual gradients) |
| **GEPAOptimizer** | Complex multi-objective agents | Evolutionary optimization with Pareto selection |
| **BayesianSearchOptimizer** | Few-shot prompt selection | Optuna-powered example selection |
| **PromptWizardOptimizer** | Creative/style-heavy prompts | Mutate, critique, refine pipeline |
| **RandomSearchOptimizer** | Baselines | Random variations |

```python
from fi.opt.optimizers import MetaPromptOptimizer  # or ProTeGi, GEPAOptimizer
from fi.opt.generators import LiteLLMGenerator
from fi.opt.base.evaluator import Evaluator
from fi.opt.datamappers import BasicDataMapper

# 1. Teacher generator
teacher = LiteLLMGenerator(model=get_litellm_model(), prompt_template="{prompt}")

# 2. Evaluator (LLM-as-a-judge)
judge = CustomLLMJudge(provider=LiteLLMProvider(), config={...}, model=model)
evaluator = Evaluator(metric=judge)

# 3. Data mapper
mapper = BasicDataMapper(key_map={"response": "generated_output", "expected_response": "answer"})

# 4. Optimize
optimizer = MetaPromptOptimizer(teacher_generator=teacher)
result = optimizer.optimize(
    evaluator=evaluator, data_mapper=mapper,
    dataset=dataset, initial_prompts=[prompt],
)
print(result.best_generator.get_prompt_template())
```

## Checklist

- [ ] `tools.py` with mock data (if applicable)
- [ ] `agent.py` with deliberately suboptimal prompt
- [ ] `agent_with_simulate.py` with AgentWrapper and test personas
- [ ] `evaluate.py` with appropriate eval templates
- [ ] `optimize.py` with matching optimizer
- [ ] `notebook.ipynb` with step-by-step walkthrough
- [ ] `README.md` documenting files, flaws, evals, and optimizer choice
