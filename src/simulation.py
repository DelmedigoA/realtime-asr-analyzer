import asyncio
from .asr_runner import run_realtime_asr, text_queue
from .analyzer import analyzer_loop, TextAnalyzer
from config import HF_PATH

async def simulation(wav, min_chunk, dict_state):
    analyzer = TextAnalyzer(HF_PATH)
    analyzer.load()
    asr_task = asyncio.create_task(run_realtime_asr(wav, min_chunk=min_chunk))
    analyzer_task = asyncio.create_task(analyzer_loop(dict_state, analyzer, text_queue))

    await asyncio.gather(asr_task, analyzer_task)
