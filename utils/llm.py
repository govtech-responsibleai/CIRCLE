"""
utils/llm.py
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Standard Library
import asyncio
import uuid
import os
from typing import Any, Callable, Awaitable, Dict, Union

# Third Party
import boto3
from anthropic import AsyncAnthropic
from botocore.exceptions import BotoCoreError
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
    If it fails for a reason besides timeout, try again up to 3 times.

    Args:
        client_method: Async function or coroutine to invoke (e.g., client.beta.messages.create).
        method_kwargs: Keyword arguments for the client method.
        timeout: Maximum time in seconds to wait for the call.
        error_prefix: Prefix for any error message returned.

    Returns:
        The model response, or an error string if the call fails or times out.
    """
    max_attempts = 3
    last_exception = None
    for attempt in range(max_attempts):
        try:
            resp = await asyncio.wait_for(client_method(**method_kwargs), timeout=timeout)
            return resp
        except asyncio.TimeoutError:
            return f"{error_prefix}: Request exceeded set timeout of {timeout} seconds."
        except Exception as e:
            last_exception = e
            # Only retry if not the last attempt
            continue
    return f"{error_prefix}: {last_exception}"


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
        The parsed model response, or an error string if the call fails or times out.
    """
    anthropic_client = AsyncAnthropic(api_key=get_env_var("ANTHROPIC_API_KEY"))

    response = await run_model(
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

    if isinstance(response, str):
        return response
    
    parsed_response = ""

    for content_block in response.content:
        if content_block.type == "text":
            parsed_response += content_block.text
        elif content_block.type == "server_tool_use" and content_block.name == "code_execution":
            parsed_response += "\n<CODE>\n"
            parsed_response += content_block.input.get("code", "")
            parsed_response += "\n```\n</CODE>\n"
        elif content_block.type == "code_execution_tool_result":
            parsed_response += "<OUTPUT>\n"
            if hasattr(content_block.content, 'stdout') and content_block.content.stdout:
                parsed_response += content_block.content.stdout
            if hasattr(content_block.content, 'stderr') and content_block.content.stderr:
                parsed_response += f"STDERR: {content_block.content.stderr}\n"
            parsed_response += "</OUTPUT>\n"

    return parsed_response


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
        The parsed model response, or an error string if the call fails or times out.
    """
    google_client = genai.Client(api_key=get_env_var("GOOGLE_API_KEY"))

    response = await run_model(
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

    if isinstance(response, str):
        return response
    
    parsed_response = ""

    if response.candidates[0].content.parts is not None:
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                parsed_response += part.text
            if part.executable_code is not None:
                parsed_response += "\n<CODE>\n"
                parsed_response += part.executable_code.code
                parsed_response += "\n```\n</CODE>\n"
            if part.code_execution_result is not None:
                parsed_response += "<OUTPUT>\n"
                parsed_response += part.code_execution_result.output
                parsed_response += "</OUTPUT>\n"

    return parsed_response



# ------------------------------------------------------------
# Mistral Model Runner
# ------------------------------------------------------------


async def run_mistral_model(
    system_prompt: str,
    prompt: str,
    model: str = "mistral-medium-latest",
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
        The parsed model response, or an error string if the call fails or times out.
    """
    mistral_client = Mistral(api_key=get_env_var("MISTRAL_API_KEY"))

    async def _mistral_call() -> Any:
        code_agent = mistral_client.beta.agents.create(
            model=model,
            name="Coding Agent",
            description="Agent used to execute code using the interpreter tool.",
            instructions=system_prompt,
            tools=[{"type": "code_interpreter"}],
            completion_args={"temperature": 0.3, "top_p": 0.95},
        )
        return await mistral_client.beta.conversations.start_async(
            agent_id=code_agent.id, inputs=prompt
        )

    resp = await run_model(
        client_method=_mistral_call,
        method_kwargs={},
        timeout=timeout,
        error_prefix="Error running Mistral model",
    )
    if isinstance(resp, str):
        return resp
    
    # print(resp)  # Optionally keep for debugging

    parsed = ""
    outputs = getattr(resp, 'outputs', [])
    for entry in outputs:
        typ = getattr(entry, 'type', None)
        if typ == 'message.output':
            parsed += getattr(entry, 'content', '')
        elif typ == 'tool.execution':
            info = getattr(entry, 'info', {})
            code = info.get('code', '')
            outp = info.get('code_output', '')
            parsed += f"\n<CODE>\n{code}\n```\n</CODE>\n"
            parsed += f"<OUTPUT>\n{outp}</OUTPUT>\n"
    return parsed


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
    resp = await run_model(
        client_method=openai_client.responses.create,
        method_kwargs={
            "model": model,
            "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
            "instructions": system_prompt,
            "input": prompt,
            "include": ["code_interpreter_call.outputs"],
        },
        timeout=timeout,
        error_prefix="Error running OpenAI model",
    )
    if isinstance(resp, str):
        return resp
    
    parsed = ""
    outputs = getattr(resp, 'output', [])
    for entry in outputs:
        typ = getattr(entry, 'type', None)
        # Handle text messages
        if typ == 'message':
            content = getattr(entry, 'content', [])
            for part in content:
                text = getattr(part, 'text', None)
                if text:
                    parsed += text
        # Handle code interpreter calls
        elif typ == 'code_interpreter_call':
            code = getattr(entry, 'code', None)
            if code:
                parsed += f"\n<CODE>\n{code}\n```\n</CODE>\n"
            outputs_list = getattr(entry, 'outputs', [])
            for out in outputs_list:
                out_type = out.get('type')
                if out_type == 'logs':
                    logs = out.get('logs', '')
                    parsed += f"<OUTPUT>\n{logs}</OUTPUT>\n"
    return parsed


# ------------------------------------------------------------
# AWS Bedrock Agent Runner
# ------------------------------------------------------------

async def run_aws_agent(
    prompt: str,
    model: str = "nova-micro",
    session_id: str = None,
    profile_name: str = None,
    timeout: float = 30.0,
) -> str:
    """
    Invoke an AWS Bedrock agent and return the completion as a string.

    Args:
        agent_id: The AWS Bedrock agent ID.
        alias_id: The AWS Bedrock agent alias ID.
        prompt: The user prompt to send to the agent.
        session_id: Optional session ID. If not provided, a random one will be generated.
        profile_name: Optional AWS profile name. If not provided, uses AWS_PROFILE env var or default.
        timeout: Timeout for the agent call (not strictly enforced, but for API symmetry).

    Returns:
        The agent's completion as a string, or an error message.
    """
    def _sync_aws_call():
        """Synchronous AWS operations to be run in executor."""
        AWS_MODELS = {
            "nova-micro": [os.getenv("NOVA_MICRO_AGENT_ID"), os.getenv("NOVA_MICRO_ALIAS_ID")], # these are already pre-configued with the system prompt on the AWS console
            "nova-lite": [os.getenv("NOVA_LITE_AGENT_ID"), os.getenv("NOVA_LITE_ALIAS_ID")],
            "nova-pro": [os.getenv("NOVA_PRO_AGENT_ID"), os.getenv("NOVA_PRO_ALIAS_ID")],
            "nova-premier": [os.getenv("NOVA_PREMIER_AGENT_ID"), os.getenv("NOVA_PREMIER_ALIAS_ID")],
        }

        actual_profile_name = profile_name if profile_name else os.getenv("AWS_PROFILE", None)
        session = boto3.Session(profile_name=actual_profile_name) if actual_profile_name else boto3.Session()
        client = session.client("bedrock-agent-runtime")
        actual_session_id = session_id if session_id else str(uuid.uuid4())
        
        response = client.invoke_agent(
            agentId=AWS_MODELS[model][0],
            agentAliasId=AWS_MODELS[model][1],
            enableTrace=True,
            sessionId=actual_session_id,
            inputText=prompt,
        )
        
        completion = ""
        for event in response.get("completion", []):
            if 'chunk' in event:
                chunk = event["chunk"]
                completion += chunk["bytes"].decode()
        return completion

    try:
        # Run the synchronous AWS operations in a thread pool
        loop = asyncio.get_event_loop()
        completion = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_aws_call),
            timeout=timeout
        )
        return completion
    except asyncio.TimeoutError:
        return f"Error running AWS agent: Request exceeded set timeout of {timeout} seconds."
    except BotoCoreError as e:
        return f"Error running AWS agent: {e}"
    except Exception as e:
        return f"Error running AWS agent: {e}"
