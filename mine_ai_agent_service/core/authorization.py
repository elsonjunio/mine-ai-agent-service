from mine_ai_agent_service.config import settings
from mine_ai_agent_service.core.utils import get_nested_claim
from mine_ai_agent_service.exceptions.application import PermissionDeniedError


def validate_role(user: dict, required_role: str) -> None:
    roles = get_nested_claim(user, settings.OPENID_ROLE_CLAIM)

    if not roles or required_role not in roles:
        raise PermissionDeniedError(f'Missing required role: {required_role}')


def is_admin(user: dict):
    roles = get_nested_claim(user, settings.OPENID_ROLE_CLAIM)

    return settings.ADMIN_ROLE in roles
