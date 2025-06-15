"""
utils/llm.py
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Standard Library
import asyncio
import os
from typing import Any, Callable, Awaitable, Dict, Union

# Third Party
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as google_types
from mistralai import Mistral
from openai import AsyncOpenAI

# ------------------------------------------------------------
# Unified Async Model Runner
# ------------------------------------------------------------


async def run_model(
    *,
    client_method: Callable[..., Awaitable[Any]],
    method_kwargs: Dict[str, Any],
    timeout: float = 30.0,
    error_prefix: str = "Error running model",
) -> Union[Any, str]:
    """
    Execute an async model call with timeout and error handling.

    Args:
        client_method: Async function or coroutine to invoke (e.g., client.beta.messages.create).
        method_kwargs: Keyword arguments for the client method.
        timeout: Maximum time in seconds to wait for the call.
        error_prefix: Prefix for any error message returned.

    Returns:
        The model response, or an error string if the call fails or times out.
    """
    try:
        resp = await asyncio.wait_for(client_method(**method_kwargs), timeout=timeout)
        return resp
    except asyncio.TimeoutError:
        return f"{error_prefix}: Request exceeded set timeout of {timeout} seconds."
    except Exception as e:
        return f"{error_prefix}: {e}"


def get_env_var(key: str) -> str:
    """
    Retrieve an environment variable or raise an error if not set.

    Args:
        key: The environment variable name.

    Returns:
        The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set.
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} is not set")
    return value


# ------------------------------------------------------------
# Anthropic Model Runner
# ------------------------------------------------------------


async def run_anthropic_model(
    system_prompt: str,
    prompt: str,
    model: str = "claude-3-5-haiku-latest",
    timeout: float = 30.0,
) -> Union[Any, str]:
    """
    Run a prompt through an Anthropic model with code execution enabled.

    Args:
        system_prompt: System-level instructions for the model.
        prompt: User prompt to send to the model.
        model: Anthropic model name.
        timeout: Maximum time in seconds to wait for the API call.

    Returns:
        The model response, or an error string if the call fails or times out.
    """
    anthropic_client = AsyncAnthropic(api_key=get_env_var("ANTHROPIC_API_KEY"))

    return await run_model(
        client_method=anthropic_client.beta.messages.create,
        method_kwargs={
            "model": model,
            "betas": ["code-execution-2025-05-22"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64_000 if "sonnet" in model.lower() else 8_192,
            "system": system_prompt,
            "tools": [{"type": "code_execution_20250522", "name": "code_execution"}],
        },
        timeout=timeout,
        error_prefix="Error running Anthropic model",
    )


# ------------------------------------------------------------
# Google Gemini (AI Studio) Model Runner
# ------------------------------------------------------------


async def run_google_model(
    system_prompt: str,
    prompt: str,
    model: str = "gemini-2.0-flash-001",
    timeout: float = 30.0,
) -> Union[Any, str]:
    """
    Run a prompt through a Google Gemini model with code execution enabled.

    Args:
        system_prompt: System-level instructions for the model.
        prompt: User prompt to send to the model.
        model: Google Gemini model name.
        timeout: Maximum time in seconds to wait for the API call.

    Returns:
        The model response, or an error string if the call fails or times out.
    """
    google_client = genai.Client(api_key=get_env_var("GOOGLE_API_KEY"))

    return await run_model(
        client_method=google_client.aio.models.generate_content,
        method_kwargs={
            "model": model,
            "contents": prompt,
            "config": google_types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[
                    google_types.Tool(code_execution=google_types.ToolCodeExecution())
                ],
            ),
        },
        timeout=timeout,
        error_prefix="Error running Google model",
    )


# ------------------------------------------------------------
# Mistral Model Runner
# ------------------------------------------------------------


async def run_mistral_model(
    system_prompt: str,
    prompt: str,
    model: str = "mistral-small-latest",
    timeout: float = 30.0,
) -> Union[Any, str]:
    """
    Run a prompt through a Mistral model with code interpreter enabled.

    Args:
        system_prompt: System-level instructions for the model.
        prompt: User prompt to send to the model.
        model: Mistral model name.
        timeout: Maximum time in seconds to wait for the API call.

    Returns:
        The model response, or an error string if the call fails or times out.
    """
    mistral_client = Mistral(api_key=get_env_var("MISTRAL_API_KEY"))

    async def _mistral_call() -> Any:
        code_agent = mistral_client.beta.agents.create(
            model=model,
            name="Coding Agent",
            description="Agent used to execute code using the interpreter tool.",
            instructions=system_prompt,
            tools=[{"type": "code_interpreter"}],
            completion_args={
                "temperature": 0.3,
                "top_p": 0.95,
            },
        )

        response = await mistral_client.beta.conversations.start_async(
            agent_id=code_agent.id, inputs=prompt
        )

        return response

    return await run_model(
        client_method=_mistral_call,
        method_kwargs={},
        timeout=timeout,
        error_prefix="Error running Mistral model",
    )


# ------------------------------------------------------------
# OpenAI Model Runner
# ------------------------------------------------------------


async def run_oai_model(
    system_prompt: str,
    prompt: str,
    model: str = "gpt-4.1-nano",
    timeout: float = 30.0,
) -> Union[Any, str]:
    """
    Run a prompt through an OpenAI model with code interpreter enabled.

    Args:
        system_prompt: System-level instructions for the model.
        prompt: User prompt to send to the model.
        model: OpenAI model name.
        timeout: Maximum time in seconds to wait for the API call.

    Returns:
        The model response, or an error string if the call fails or times out.
    """
    openai_client = AsyncOpenAI(api_key=get_env_var("OPENAI_API_KEY"))

    return await run_model(
        client_method=openai_client.responses.create,
        method_kwargs={
            "model": model,
            "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
            "instructions": system_prompt,
            "input": prompt,
        },
        timeout=timeout,
        error_prefix="Error running OpenAI model",
    )
