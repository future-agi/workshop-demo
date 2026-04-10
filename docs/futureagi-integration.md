# FutureAGI Integration Guide

How the Simulate SDK, evaluation, and optimization work together.

## Architecture

```
                    Your Agent
                        |
                   AgentWrapper        <-- Wrap your agent
                        |
                   TestRunner          <-- Run simulated conversations
                        |
                    Scenario           <-- Define test personas
                        |
                   TestReport          <-- Get transcripts
                        |
               evaluate_report()       <-- Score with eval templates
                        |
                  EvaluationScores     <-- Identify weaknesses
                        |
                    Optimizer          <-- Improve the prompt
                        |
                 Optimized Prompt      <-- Better agent
```

## Simulate SDK (`agent-simulate`)

### Wrapping Your Agent

Every agent needs an `AgentWrapper` that implements `async call()`:

```python
from fi.simulate import AgentWrapper, AgentInput, AgentResponse

class MyWrapper(AgentWrapper):
    async def call(self, input: AgentInput) -> AgentResponse:
        # input.messages = conversation history [{role, content}, ...]
        # input.new_message = the latest message
        # input.thread_id = conversation identifier
        
        response = your_llm_call(input.messages)
        
        return AgentResponse(
            content=response_text,
            tool_calls=tool_calls_list,       # optional
            tool_responses=tool_results_list,  # optional
            metadata={"model": "..."},         # optional
        )
```

Built-in wrappers:
- `OpenAIAgentWrapper` - For OpenAI SDK
- `AnthropicAgentWrapper` - For Anthropic SDK
- `GeminiAgentWrapper` - For Google Gemini
- `LangChainAgentWrapper` - For LangChain AgentExecutor

### Test Scenarios

```python
from fi.simulate import Scenario, Persona

scenario = Scenario(
    name="my-test",
    dataset=[
        Persona(
            persona={"name": "Alice", "age": 30, "mood": "curious"},
            situation="She wants to learn about X.",
            outcome="Clear explanation of X.",
        ),
    ],
)
```

### Running Tests

**Cloud mode** (for chat/text agents):
```python
runner = TestRunner()  # reads FI_API_KEY from env
report = await runner.run_test(
    run_test_name="my-test",
    agent_callback=wrapper,
    scenario=scenario,
)
```

**Local mode** (for LiveKit voice agents):
```python
agent_def = AgentDefinition(name="...", url=LIVEKIT_URL, room_name="...", ...)
report = await runner.run_test(
    agent_definition=agent_def,
    scenario=scenario,
    simulator=simulator_def,
    record_audio=True,
)
```

## Evaluation

### Built-in Templates (79 available)

FutureAGI provides 79 evaluation templates. Key ones by category:

**General Quality:**
- `task_completion` - Did it fulfill the request?
- `is_helpful` - Was the response useful?
- `is_concise` - Was it brief and clear?
- `tone` - Was the tone appropriate?
- `groundedness` - Is it grounded in provided context?
- `detect_hallucination` - Did it make things up?

**Tool/Function Calling:**
- `evaluate_function_calling` - Were tools called correctly?

**Customer Agent (conversational):**
- `conversation_coherence` - Logical conversation flow
- `conversation_quality` - Overall conversation quality
- `context_retention` - Remembered details across turns
- `query_handling` - Handled queries effectively
- `clarification_seeking` - Asked for clarification when needed
- `loop_detection` - Avoided repeating itself
- `objection_handling` - Handled objections well
- `human_escalation` - Escalated appropriately
- `prompt_conformance` - Followed system prompt

**Safety:**
- `toxicity`, `pii_detection`, `bias_detection`, `prompt_injection`
- `no_llm_reference` - Avoided "As an AI..."
- `no_racial_bias`, `no_gender_bias`, `no_age_bias`

**Audio/Voice:**
- `audio_transcription` - STT accuracy
- `audio_quality` - Audio clarity
- `tts_accuracy` - TTS naturalness

**Statistical:**
- `bleu_score`, `rouge_score`, `embedding_similarity`

### Usage

```python
from fi.simulate import evaluate_report

report = evaluate_report(
    report,
    eval_templates=["task_completion", "is_helpful", "is_concise"],
    model_name="turing_flash",
)

for result in report.results:
    for template, scores in result.evaluation.items():
        print(f"{template}: {scores['score']} - {scores['reason']}")
```

## Optimization (`agent-opt`)

### How It Works

1. **Generator**: Fills a prompt template with input vars and calls LLM
2. **Evaluator**: Scores the output (heuristic, LLM judge, or FutureAGI platform)
3. **Data Mapper**: Maps generator output + dataset fields to evaluator inputs
4. **Optimizer**: Iteratively improves the prompt using the evaluator feedback

### Choosing an Optimizer

| Situation | Optimizer | Why |
|-----------|-----------|-----|
| Simple chat agent, single prompt | **MetaPromptOptimizer** | Evaluates, hypothesizes, rewrites entire prompt |
| Agent with specific failures (wrong tools, missing steps) | **ProTeGi** | Beam search + targeted critiques finds specific fixes |
| Complex agent, multiple competing objectives | **GEPAOptimizer** | Evolutionary approach balances multiple goals |
| Need optimal few-shot examples | **BayesianSearchOptimizer** | Optuna finds best example combinations |
| Creative/style optimization | **PromptWizardOptimizer** | Mutation + critique + refinement pipeline |
| Baseline comparison | **RandomSearchOptimizer** | Random variations for comparison |

### Evaluation Options for Optimization

**Option A: LLM-as-a-Judge (local)**
```python
from fi.evals.metrics import CustomLLMJudge
from fi.evals.llm import LiteLLMProvider

judge = CustomLLMJudge(
    provider=LiteLLMProvider(),
    config={"name": "judge", "grading_criteria": "Score 0-1 on quality..."},
    model="groq/llama-3.3-70b-versatile",
)
evaluator = Evaluator(metric=judge)
```

**Option B: Heuristic metrics**
```python
from fi.evals.metrics import BLEUScore
evaluator = Evaluator(metric=BLEUScore())
```

**Option C: FutureAGI platform templates**
```python
evaluator = Evaluator(
    eval_template="summary_quality",
    eval_model_name="turing_flash",
)
```

## Tips

1. **Start simple**: Use MetaPromptOptimizer first, graduate to ProTeGi/GEPA for complex agents
2. **Design for evaluation**: Write prompts with specific, fixable flaws so evaluation has something to catch
3. **Match evals to agent type**: Chat agents need different evals than voice agents
4. **Dataset quality matters**: Optimization is only as good as your dataset examples
5. **Use LiteLLM model format**: `"groq/model-name"`, `"openai/model-name"`, etc.
