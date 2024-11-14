'''
스텝 2. Fastapi 기능 구현하기
데이터 입력 -> 모델 -> 결과 출력

>> uvicorn main:app --reload
'''

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models.image_model import CNNModel

import uuid
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")

@app.get("/")
def read_root():
    return {"hello" : "world"}

@app.get("/web", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="./index.html"
    )

@app.post("/upload")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "app/input"  # 이미지를 저장할 서버 경로

    files = os.listdir(UPLOAD_DIR)
    for f in files:
        os.remove(UPLOAD_DIR+'/'+f)
    
    content = await file.read()
    filename = f"{str(uuid.uuid4())}.jpg"  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
    
    return {'message': f"Successfuly uploaded {UPLOAD_DIR+'/'+filename}"}

@app.post("/predict")
async def predict_photo():
    model = CNNModel()
    result_tuple = model.predict()

    
    return {'class': str(result_tuple[0]), 
            'predict': str(result_tuple[1]), 
            'accuracy': str(result_tuple[2]),
            'image': str(model.file_path)
            }