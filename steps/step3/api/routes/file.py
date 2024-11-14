from fastapi import APIRouter, UploadFile
import uuid
import os

router = APIRouter()

@router.post("/upload")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "./storage"  # 이미지를 저장할 서버 경로

    files = os.listdir(UPLOAD_DIR)
    for f in files:
        os.remove(UPLOAD_DIR+'/'+f)
    
    content = await file.read()
    filename = f"{str(uuid.uuid4())}.jpg"  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
    
    return {'message': f"Successfuly uploaded {UPLOAD_DIR+'/'+filename}"}