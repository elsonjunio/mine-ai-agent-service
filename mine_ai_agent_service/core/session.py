from jose import jwt
from datetime import datetime, timedelta
from mine_ai_agent_service.config import settings


def decode_internal_token(token: str):
    return jwt.decode(
        token,
        settings.INTERNAL_TOKEN_SECRET,
        algorithms=['HS256'],
    )
