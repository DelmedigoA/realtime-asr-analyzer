import re
import numpy as np
import librosa
from config import SAMPLING_RATE

def strip_timestamps(line: str) -> str:
    m = re.match(r"^\s*\d+(?:\.\d+)?\s+\d+\s+\d+\s+(.*)$", line)
    return m.group(1).strip() if m else line.strip()

def load_audio(fname):
    a, _ = librosa.load(fname, sr=SAMPLING_RATE, dtype=np.float32)
    return a

def load_audio_chunk(audio, beg, end):
    beg_s = int(beg * SAMPLING_RATE)
    end_s = int(end * SAMPLING_RATE)
    return audio[beg_s:end_s]
