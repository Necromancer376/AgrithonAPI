from fastapi import FastAPI, Response
from pydantic import BaseModel
import json
import io
from starlette.responses import StreamingResponse
from PIL import Image

app = FastAPI()

#db = []


@app.get('/')
def recieve_messages():
    res = {}
    with open('./log/convert.txt',mode='r') as myfile:
        for j,i in enumerate(myfile.readlines()):
            res[j] = json.loads(i)
            with open("./log/"+res[j]["img"], "rb") as image:
                f = image.read()
                b = bytearray(f)
                new_str = "".join(map(chr, b))

                res[j]["img"] = new_str

    return res