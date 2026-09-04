"""FastAPI 应用装配：CORS、异常处理、路由挂载、启动初始化。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api.router import api_router
from .core.config import settings
from .core.exceptions import AppError, ErrorCode
from .core.logging import get_logger
from .db.init_db import ensure_agents_registered, init_db
from .services.async_jobs import start_job_worker, stop_job_worker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: init database + seed + agents")
    await init_db()
    ensure_agents_registered()
    start_job_worker()
    logger.info("startup: done")
    yield
    await stop_job_worker()
    logger.info("shutdown: bye")


app = FastAPI(
    title="HR 行政 Harness API",
    description="中小企业 HR/行政 智能助手（RAG + 多 Agent + 全模态对话）",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_PREFIX)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.http_status,
        content={"code": exc.code, "message": exc.message, "data": exc.data},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"loc": list(item.get("loc", [])), "msg": item.get("msg", ""), "type": item.get("type", "")}
        for item in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "code": ErrorCode.VALIDATION_ERROR,
            "message": "请求参数校验失败",
            "data": {"errors": errors},
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"code": 50000, "message": "服务器内部错误", "data": None},
    )


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": __version__,
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/v1/system/health",
    }
