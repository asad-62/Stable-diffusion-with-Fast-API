from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List  # Import List from typing
from ml import obtain_image
app=FastAPI()
@app.get("/")
def read_root():
    return {"Hello":"World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id,"message":"Hello World"}

class Item(BaseModel):
    name:str
    price: float
    tags: List[str] = []

@app.post("/items")
def create_items(Item: Item):
    return Item
@app.get("/generate")
def generate_image(prompt:str):
    image=obtain_image(prompt,num_inference_steps=5,seed=1024)
    image.save("generated_image.png")
    return FileResponse("generated_image.png") 


