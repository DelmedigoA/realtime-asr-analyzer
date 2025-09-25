from transformers import AutoTokenizer, AutoModelForCausalLM
from ast import literal_eval
import torch
from config import ROLE, SHOTS, COMMAND

class TextAnalyzer:
    def __init__(self, hf_path):
        self.hf_path = hf_path

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_path, dtype=torch.float16).to("cuda")

    def __call__(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    @staticmethod
    def get_prompt(role="", few_shots="", dict_state={}, command=""):
        prompt = f"""{role}\n{few_shots}\n{command}"""
        return prompt

async def analyzer_loop(dict_state, text_analyzer, text_queue):
    while True:
        full_text = await text_queue.get()
        if full_text is None:
            break

        prompt = text_analyzer.get_prompt(
          role = ROLE, few_shots = SHOTS, command = COMMAND.format(
          full_text=full_text,
          dict_state=dict_state))
        raw_output = text_analyzer(prompt).strip().split("<|im_end|>")[0]
        print("🔎 LLM Decisions:", raw_output)
        try:
            decisions = literal_eval(raw_output)
            if isinstance(decisions, list):
                for d in decisions:
                    exec(d)
        except Exception as e:
            print(f"⚠️ Could not parse model output: {raw_output} ({e})")

        print("📊 Updated dict_state:", dict_state)
        text_queue.task_done()
