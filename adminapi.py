from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

#db = []

class Text(BaseModel):
    nickname:str
    text:str


@app.get('/')
def recieve_messages():
    with open('chat.txt',mode='r') as myfile:
        return [json.loads(i) for i in myfile.readlines()]