from fastapi import APIRouter, Depends
from pydantic import BaseModel

from mine_ai_agent_service.api.dependencies.auth import CurrentUser, get_current_user
from mine_ai_agent_service.services.agent_service import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post('/chat')
async def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    result = await run_agent(request=body.message, token=user.token)
    return ChatResponse(response=result)