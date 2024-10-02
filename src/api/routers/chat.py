from typing import Annotated, Final

from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, SystemMessage
from api.setting import DEFAULT_MODEL
import logging

router = APIRouter(
    prefix="/{deployment}/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)

JSON_MODE_SYSTEM_PROMPT: Final = SystemMessage(
    name="json_mode",
    role="system",
    content="You are a helpful assistant designed to output JSON without extra text",
)


@router.post(
    "/completions",
    response_model=ChatResponse | ChatStreamResponse,
    response_model_exclude_unset=True,
)
async def chat_completions(
    deployment: str,
    chat_request: Annotated[
        ChatRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                }
            ],
        ),
    ],
):
    if not deployment or deployment == "" or deployment.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL
    else:
        chat_request.model = deployment

    # JSON mode handling
    # Ref: https://platform.openai.com/docs/guides/structured-outputs/json-mode
    response_format = chat_request.response_format
    if (
        chat_request.stream
        or response_format
        and response_format["type"] == "json_object"
    ):
        chat_request.messages.append(JSON_MODE_SYSTEM_PROMPT)

    logging.debug(f"chat_request: {chat_request}")

    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)
    if chat_request.stream:
        return StreamingResponse(
            content=model.chat_stream(chat_request), media_type="text/event-stream"
        )
    return model.chat(chat_request)
