from jose import jwt
from mine_backend.config import settings
from mine_backend.core.keycloak import get_jwks
from mine_backend.services.session_service import decode_internal_token
from mine_backend.exceptions.application import (
    InvalidTokenError,
    STSCredentialsNotFoundError,
)


# --- Token Keycloak


async def verify_keycloak_token(token: str) -> dict:
    jwks = await get_jwks()

    try:
        payload = jwt.decode(
            token,
            jwks,
            algorithms=['RS256'],
            audience=settings.KEYCLOAK_CLIENT_ID,
        )
        return payload

    except Exception:
        raise InvalidTokenError('Invalid token')


# --- Token interno


def verify_internal_session(token: str) -> dict:
    try:
        return decode_internal_token(token)
    except Exception:
        raise InvalidTokenError('Invalid internal token')


def extract_sts_credentials(session: dict) -> dict:
    sts = session.get('sts')

    if not sts:
        raise STSCredentialsNotFoundError('STS credentials not found')

    return {
        'aws_access_key_id': sts['access_key'],
        'aws_secret_access_key': sts['secret_key'],
        'aws_session_token': sts.get('session_token'),
    }
