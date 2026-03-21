from dataclasses import dataclass

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from mine_ai_agent_service.core.session import decode_internal_token

security = HTTPBearer()


@dataclass
class CurrentUser:
    token: str
    payload: dict


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CurrentUser:
    token = credentials.credentials
    payload = decode_internal_token(token)
    return CurrentUser(token=token, payload=payload)