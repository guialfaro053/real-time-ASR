import asyncio
import numpy as np
import uvicorn
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware as CORS
from starlette.websockets import WebSocketDisconnect
from typing import Annotated
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Query,
    WebSocket,
    Request
)
import uuid
from typing import Annotated
from starlette.websockets import WebSocketDisconnect
import random
import string
import time
import uuid
import logging

logger_main = logging.getLogger('main')
logger_main.setLevel(logging.DEBUG)
logger_main.handlers = []  
logger_main.propagate = False
 
from dotenv import load_dotenv
load_dotenv()
 
app = FastAPI()

origins = ["*"]
app.add_middleware(CORS,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

MODEL_TYPE = "small.en" 
LANGUAGE_CODE = "en"

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
 
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
 
    def disconnect(self, user_id: str):
        del self.active_connections[user_id]
        print(self.active_connections)
        
    async def send_personal_json(self, message: str, user_id: str):
        await self.active_connections[user_id].send_json(message)

async def get_cookie_or_token(
    websocket: WebSocket,
    session: Annotated[str | None, Cookie()] = None,
    token: Annotated[str | None, Query()] = None,
):
    if session is None and token is None:
        return str(uuid.uuid4())
    return session or token

def create_whisper_model() -> WhisperModel:
    whisper = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8", num_workers=2, cpu_threads=8)
    print("Loaded model")
    return whisper

model = create_whisper_model()
manager = ConnectionManager()

async def parse_body(request: Request):
    data: bytes = await request.body()
    return data

def execute_blocking_whisper_prediction(model: WhisperModel, audio_data_array) -> str:
    segments, _ = model.transcribe(audio_data_array,
                                   language=LANGUAGE_CODE,
                                   beam_size=5,
                                   vad_filter=True,
                                   vad_parameters=dict(min_silence_duration_ms=1000),
                                   log_prob_threshold=-0.9,
                                   no_speech_threshold=0.08,
                                   initial_prompt="")
    
    segments = [s.text for s in segments]
    transcription = " ".join(segments)
    transcription = transcription.strip()
    return transcription

async def random_user_id():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    current_time = str(int(time.time()))
    return f"{str(uuid.uuid4())}-{random_string}-{current_time}"

@app.websocket("/transcribe")    
async def connect_websocket(websocket: WebSocket):
    user_id = await random_user_id()
    await manager.connect(websocket, user_id)
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            audio_data_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0

           
            result = await asyncio.get_running_loop().run_in_executor(
                None, execute_blocking_whisper_prediction, model, audio_data_array
            )
            if result != "":
                data_result = {
                    "output_data": result,
                    "client_id": user_id,
                }
                print(data_result)
                await manager.send_personal_json(data_result, user_id)
                
    except WebSocketDisconnect as e:
        manager.disconnect(user_id)
        print(f"Error: {e}")
        code = e.code
        print(f"Websocket disconnected with code {code}")

            
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
