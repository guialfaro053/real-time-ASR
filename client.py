import asyncio
import queue
import speech_recognition as sr
from datetime import datetime, timedelta
import websockets
import json
import audioop

def is_silent(audio_data, threshold=500):
    rms = audioop.rms(audio_data, 2) 
    return rms < threshold

async def main():
    SERVER_WS_URL = f"ws://localhost:8008/transcribe"
    
    vars = {"record_timeout": 2.4, "phrase_timeout": 3}
    data_queue = queue.Queue()
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000

    microphone = sr.Microphone(sample_rate=16000)

    def record_callback(recognizer, audio):
        data_queue.put(audio.get_raw_data())
        print("Audio has been queued.")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
    recognizer.listen_in_background(microphone, record_callback, phrase_time_limit=vars["record_timeout"])

    last_sample = bytes()
    phrase_time = None
    max_buffer_size = 105000 
    condition_to_end_session = False

    async with websockets.connect(SERVER_WS_URL) as websocket:
        try:
            while True:
                now = datetime.utcnow()
                buffer_full = len(last_sample) >= max_buffer_size

                if not data_queue.empty():
                    phrase_time = now
                    while not data_queue.empty():
                        last_sample += data_queue.get()

                if last_sample and (buffer_full or (phrase_time and now - phrase_time > timedelta(seconds=vars["phrase_timeout"]))):
                    audio_data = sr.AudioData(last_sample, microphone.SAMPLE_RATE, microphone.SAMPLE_WIDTH).get_wav_data()

                    if not is_silent(audio_data):
                        await websocket.send(audio_data)

                        response = await websocket.recv()
                        break
                    #     result = json.loads(response)
                    #     if result.get("text"):
                    #         print("Transcription:", result.get("text"))
                    #         condition_to_end_session = True
                    #     else:
                    #         print("No transcription received.")
                    # else:
                    #     print("Skipping silent audio.")

                    # last_sample = bytes() 
                    # phrase_time = None

                    # if condition_to_end_session:
                    #     print(result)
                    #     return result
                    #     break
                    
            # await websocket.close()

        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    asyncio.run(main())

