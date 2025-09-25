import asyncio, time
import numpy as np
from IPython.display import clear_output
from whisper_streaming.whisper_online import FasterWhisperASR, OnlineASRProcessor
from config import LANGUAGE, MODEL_SIZE, SAMPLING_RATE
from .utils import load_audio, load_audio_chunk, strip_timestamps

text_queue = asyncio.Queue()

asr = FasterWhisperASR(LANGUAGE, MODEL_SIZE, model_dir = "/content/realtime-asr-analyzer/whisper_dir")
online = OnlineASRProcessor(asr)

silence = np.zeros(SAMPLING_RATE, dtype=np.float32)
asr.transcribe(silence)
online.insert_audio_chunk(silence)
_ = online.process_iter()
print("✅ Model ready and warmed up!")

async def run_realtime_asr(audio_path, min_chunk=1.0):
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLING_RATE

    beg, start = 0.0, time.time()
    transcription = ""

    while beg < duration:
        end = min(beg + min_chunk, duration)
        chunk = load_audio_chunk(audio, beg, end)
        online.insert_audio_chunk(chunk)

        o = online.process_iter()
        if o[0] is not None:
            now = time.time() - start
            text = strip_timestamps(
                f"{now*1000:.4f} {o[0]*1000:.0f} {o[1]*1000:.0f} {o[2]}"
            )
            transcription += text + " "
            await text_queue.put(transcription)
            print(transcription)

        beg = end
        await asyncio.sleep(min_chunk)

    o = online.finish()
    if o[0] is not None:
        text = strip_timestamps(
            f"{o[0]*1000:.0f} {o[1]*1000:.0f} {o[2]}"
        )
        transcription += text
        await text_queue.put(transcription)

    clear_output(wait=True)
    print(transcription)
