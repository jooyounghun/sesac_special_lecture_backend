'''
스텝 3. Fastapi 구조 코딩하기
FastAPI 개발자는 FastAPI를 어떻게 쓸까?
https://github.com/fastapi/full-stack-fastapi-template/tree/master/backend/app

ex)
- main.py # FastAPI 애플리케이션의 진입점이 되는 파일로, API 라우터를 구성하고 애플리케이션을 실행
- models.py # 모델이 정의된 파일
- api # FastAPI 애플리케이션의 API 엔드포인트 및 라우터가 정의된 디렉토리
- core # 애플리케이션의 핵심 기능 및 설정이 정의된 모듈이 위치하는 디렉토리
- tests # 테스트 파일이 위치한 디렉토리
- schemas # Pydantic 스키마 정의가 포함된 디렉토리로, 데이터의 유효성 검사와 API 요청 및 응답의 구조를 정의
- backend_pre_start.py # FastAPI 애플리케이션 시작 전에 실행할 코드가 정의된 파일
>> uvicorn main:app --reload
'''

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.main import api_router
from core.config import settings

# if settings.ENVIRONMENT != "local":
#     sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.mount(settings.API_V1_STR+"/static", StaticFiles(directory="frontend/static"), name="static")

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
