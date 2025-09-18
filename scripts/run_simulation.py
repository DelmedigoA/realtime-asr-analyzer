import asyncio
import sys
from src.simulation import simulation

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_simulation.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    dict_state = {"first_name": None, "last_name": None, "age": None}

    asyncio.run(simulation(audio_path, min_chunk=1.0, dict_state=dict_state))
