from fastapi import APIRouter

from mine_ai_agent_service.api.routers import chat

api_router = APIRouter()

api_router.include_router(chat.router)
