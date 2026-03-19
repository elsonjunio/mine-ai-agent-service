from mine_ai_agent_service.exceptions.base import AppException


class InvalidTokenError(AppException):
    def __init__(self, message: str = 'Invalid token'):
        super().__init__(message, code='INVALID_TOKEN')


class STSCredentialsNotFoundError(AppException):
    def __init__(self, message: str = 'STS credentials not found'):
        super().__init__(message, code='STS_CREDENTIALS_NOT_FOUND')


class PermissionDeniedError(AppException):
    def __init__(self, message: str = 'Permission denied'):
        super().__init__(message, code='PERMISSION_DENIED')


class InconsistentDataError(AppException):
    def __init__(self, message: str = 'Inconsistent fields/data'):
        super().__init__(message, code='INCONSISTENT_DATA')


class NotFoundError(AppException):
    def __init__(self, message: str = 'The specified was not found'):
        super().__init__(message, code='NOT_FOUND')


class AlreadyExistsError(AppException):
    def __init__(self, message: str = 'Already exists'):
        super().__init__(message, code='ALREADY_EXISTS')


class UnexpectedError(AppException):
    def __init__(self, message: str = 'Unexpected error'):
        super().__init__(message, code='UNEXPECTED_ERROR')


class ServiceUnavailableError(AppException):
    def __init__(self, message: str = 'Service unavailable'):
        super().__init__(message, code='UNAVAILABLE_ERROR')
