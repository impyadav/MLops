"""
# -*- coding: utf-8 -*-

Created on Jan 2022
@author: Prateek Yadav

"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.modelling import inference

app = FastAPI()


class Item(BaseModel):
    ip_word: str
    n: int


@app.get('/')
async def root():
    return {'messgae': 'Hello world!'}


@app.get('/predictions/')
async def get_next_word(item: Item):
    infer_obj = inference.Inference(item.ip_word, item.n)
    return {'predictions': infer_obj.get_nearest_neighbors()}
