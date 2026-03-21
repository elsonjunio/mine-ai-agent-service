from fastapi.responses import JSONResponse
from fastapi import Request
from starlette.responses import Response
from mine_ai_agent_service.exceptions.base import AppException
from typing import Dict
import logging


ERROR_STATUS_MAP: Dict[str, int] = {
    'INVALID_TOKEN': 401,
    'STS_CREDENTIALS_NOT_FOUND': 401,
    'PERMISSION_DENIED': 403,
    'INCONSISTENT_DATA': 400,
    'BUCKET_NOT_FOUND': 404,
    'BUCKET_ALREADY_EXISTS': 409,
    'BUCKET_UNEXPECTED_ERROR': 502,
    'UNEXPECTED_ERROR': 500,

    'GROUP_NOT_FOUND': 404,
    'GROUP_ALREADY_EXISTS' : 409,

    'UNAVAILABLE_ERROR': 503
}


async def app_exception_handler(
    request: Request,
    exc: Exception,
) -> Response:
    assert isinstance(exc, AppException)

    status_code = ERROR_STATUS_MAP.get(exc.code, 400)

    return JSONResponse(
        status_code=status_code,
        content={
            'success': False,
            'data': None,
            'error': {
                'code': exc.code,
                'message': exc.message,
            },
        },
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> Response:

    logging.exception('Unhandled exception occurred')

    return JSONResponse(
        status_code=500,
        content={
            'success': False,
            'data': None,
            'error': {
                'code': 'INTERNAL_SERVER_ERROR',
                'message': f'Unexpected error occurred. {exc}',
            },
        },
    )
