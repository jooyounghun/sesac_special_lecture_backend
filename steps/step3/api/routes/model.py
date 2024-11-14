from fastapi import APIRouter
from core.image_model import CNNModel

router = APIRouter()

@router.post("/predict")
async def predict_photo():
    model = CNNModel()
    result_tuple = model.predict()

    
    return {'class': str(result_tuple[0]), 
            'predict': str(result_tuple[1]), 
            'accuracy': str(result_tuple[2]),
            'image': str(model.file_path)
            }