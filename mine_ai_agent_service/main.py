from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mine_ai_agent_service.api.router import api_router
from mine_ai_agent_service.core.logging_config import setup_logger



from mine_ai_agent_service.api.exception_handlers import (
    app_exception_handler,
    unhandled_exception_handler,
)
from mine_ai_agent_service.exceptions.base import AppException

import logging

setup_logger('DEBUG')


app = FastAPI(title='Mine Agent AI Backend')

origins = [
    "http://localhost:4200",   # Angular dev
    "http://localhost:3000",   # React dev
    "https://meudominio.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(api_router)
