from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    @app.post("/generate-description")
    async def generate_description(file: UploadFile = File(...)):
        image = Image.open(file.file).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated Caption: {caption}")
        return {"description": caption}
    uvicorn.run(app, host="0.0.0.0", port=8000)
