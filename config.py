"""
Shared LLM configuration for all workshop agents.

Reads LLM_PROVIDER and MODEL_NAME from environment variables and provides
helpers to get the right client for each framework (raw SDK, LangChain, LiteLLM).

Usage:
    from config import get_llm_client, get_langchain_llm, get_litellm_model, get_model_name
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")


def get_model_name() -> str:
    """Return the configured model name."""
    return MODEL_NAME


def get_llm_client():
    """
    Return an OpenAI-compatible chat client for the configured provider.

    All three providers (Groq, OpenAI, Anthropic) expose an OpenAI-compatible
    chat completions interface, so the calling code can use the same API:

        client = get_llm_client()
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[...]
        )
    """
    if LLM_PROVIDER == "groq":
        from groq import Groq
        return Groq()
    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI()
    elif LLM_PROVIDER == "anthropic":
        # Use Anthropic's OpenAI-compatible endpoint
        from openai import OpenAI
        return OpenAI(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1/",
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Use groq, openai, or anthropic.")


def get_langchain_llm(**kwargs):
    """
    Return a LangChain chat model for the configured provider.

    Used by the bike_rental agent which relies on LangChain's agent framework.

        llm = get_langchain_llm(temperature=0.7)
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=MODEL_NAME, **kwargs)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODEL_NAME, **kwargs)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=MODEL_NAME, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Use groq, openai, or anthropic.")


def get_litellm_model() -> str:
    """
    Return the LiteLLM model string for the configured provider.

    Used by agent-opt optimizers which use LiteLLM under the hood.
    Format: "provider/model-name" (e.g., "groq/llama-3.3-70b-versatile")

    For OpenAI, LiteLLM accepts the model name directly without prefix.
    """
    if LLM_PROVIDER == "openai":
        return MODEL_NAME
    return f"{LLM_PROVIDER}/{MODEL_NAME}"


def get_livekit_llm_plugin():
    """
    Return a configured LiveKit LLM plugin for the configured provider.

    LiveKit uses its own openai plugin that can point to different base URLs.
    Used by the hotel_voice agent.
    """
    from livekit.plugins import openai as lk_openai

    if LLM_PROVIDER == "groq":
        return lk_openai.LLM(
            model=MODEL_NAME,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif LLM_PROVIDER == "openai":
        return lk_openai.LLM(model=MODEL_NAME)
    elif LLM_PROVIDER == "anthropic":
        # LiveKit's openai plugin can talk to Anthropic's openai-compatible endpoint
        return lk_openai.LLM(
            model=MODEL_NAME,
            base_url="https://api.anthropic.com/v1/",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}.")
