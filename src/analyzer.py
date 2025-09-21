from transformers import AutoTokenizer, AutoModelForCausalLM
from ast import literal_eval
import torch

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
        prompt = f"""{role}\n{dict_state}\n{few_shots}\n{command}"""
        return prompt

async def analyzer_loop(dict_state, text_analyzer, text_queue):
    while True:
        full_text = await text_queue.get()
        if full_text is None:
            break

        prompt = text_analyzer.get_prompt(
            role=(
                "You are a survey dict_state updater.\n"
                "You ALWAYS try to extract useful structured information from the transcript.\n"
                "If the transcript contains a name, age, address, date, or other respondent info, "
                "you MUST update dict_state accordingly.\n"
                "If nothing new is found, return [].\n\n"
                "Rules:\n"
                "  • Output ONLY a valid Python list of assignment strings.\n"
                "  • Each assignment must be of the form: \"dict_state['field'] = value\".\n"
                "  • Values must be valid Python (strings quoted, numbers as ints).\n\n"
                "Examples:\n"
                "Text: 'ron is 35 years old.'\n"
                "dict_state = {'respondent_first_name': 'ron', 'age': None}\n"
                "Decisions: [\"dict_state['age'] = 35\"]\n\n"
                "Text: 'שלום, קוראים לי יעל ואני ממש שמחה לדבר איתך'\n"
                "dict_state = {'respondent_first_name': 'יעל'}\n"
                "Decisions: []\n\n"
                "--- Complex Examples ---\n"
                "Example 1 (already updated, no decisions):\n"
                "Text: 'שלום, קוראים לי דנה כהן, אני בת 28 וגרה ברחוב הראשי 15.'\n"
                "dict_state = {'respondent_first_name': 'דנה', 'respondent_last_name': 'כהן', 'age': 28, 'full_address': 'רחוב הראשי 15'}\n"
                "Decisions: []\n\n"
                "Example 2 (correction):\n"
                "Text: 'סליחה, אני בת 29 ולא 28.'\n"
                "dict_state = {'respondent_first_name': 'דנה', 'age': 28}\n"
                "Decisions: [\"dict_state['age'] = 29\"]\n\n"
                "Example 3 (many decisions at once):\n"
                "Text: 'שמי רון לוי, אני בן 42. אני גר ברחוב הגפן 10. החוזה התחיל ב-1.1.2023 ואני משלם 4500 ש\"ח לחודש על דירה של 70 מטר.'\n"
                "dict_state = {'respondent_first_name': None, 'respondent_last_name': None, 'age': None, 'full_address': None, 'contract_start_date_in_date_format': None, 'amount_paid': None, 'currency': None, 'apartment_size_sq': None}\n"
                "Decisions: [\"dict_state['respondent_first_name'] = 'רון'\", \"dict_state['respondent_last_name'] = 'לוי'\", \"dict_state['age'] = 42\", \"dict_state['full_address'] = 'רחוב הגפן 10'\", \"dict_state['contract_start_date_in_date_format'] = '2023-01-01'\", \"dict_state['amount_paid'] = 4500\", \"dict_state['currency'] = 'ש\\\"ח'\", \"dict_state['apartment_size_sq'] = 70\"]\n\n"
                "Example 4 (no decision):\n"
                "Text: 'אני לא זוכר כרגע, נדבר על זה בפעם אחרת.'\n"
                "dict_state = {'respondent_first_name': 'משה'}\n"
                "Decisions: []\n\n"
                "Example 5 (very complex):\n"
                "Text: 'אני ענת פרידמן, נולדתי ב-15 במרץ 1990, כלומר אני בת 35. אני גרה ברחוב הים 22. נכנסתי לדירה ב-15.07.2024, אני משלמת 3200 דולר לחודש על דירה של 85 מטר.'\n"
                "dict_state = {'respondent_first_name': None, 'respondent_last_name': None, 'age': None, 'birth_date': None, 'full_address': None, 'contract_start_date_in_date_format': None, 'amount_paid': None, 'currency': None, 'apartment_size_sq': None}\n"
                "Decisions: [\"dict_state['respondent_first_name'] = 'ענת'\", \"dict_state['respondent_last_name'] = 'פרידמן'\", \"dict_state['birth_date'] = '1990-03-15'\", \"dict_state['age'] = 35\", \"dict_state['full_address'] = 'רחוב הים 22'\", \"dict_state['contract_start_date_in_date_format'] = '2024-07-15'\", \"dict_state['amount_paid'] = 3200\", \"dict_state['currency'] = 'דולר'\", \"dict_state['apartment_size_sq'] = 85\"]\n\n"
                "Example 6 (EN, multiple fields + normalized date):\n"
                "Text: 'My name is Sarah Ben-David, I am 31. Address: 12 Oak Street. Lease started on 2022/11/05. I pay $1200 per month for a 60 sqm apartment.'\n"
                "dict_state = {'respondent_first_name': None, 'respondent_last_name': None, 'age': None, 'full_address': None, 'contract_start_date_in_date_format': None, 'amount_paid': None, 'currency': None, 'apartment_size_sq': None}\n"
                "Decisions: [\"dict_state['respondent_first_name'] = 'Sarah'\", \"dict_state['respondent_last_name'] = 'Ben-David'\", \"dict_state['age'] = 31\", \"dict_state['full_address'] = '12 Oak Street'\", \"dict_state['contract_start_date_in_date_format'] = '2022-11-05'\", \"dict_state['amount_paid'] = 1200\", \"dict_state['currency'] = 'USD'\", \"dict_state['apartment_size_sq'] = 60\"]\n\n"
                "Example 7 (HE, תאריכים מילוליים + EUR):\n"
                "Text: 'קוראים לי איתמר נבון, בן 33. אני גר ברחוב הדקל 5. החוזה התחיל ב-1 בספטמבר 2021. התשלום הוא 1800 יורו לחודש על 55 מטר.'\n"
                "dict_state = {'respondent_first_name': None, 'respondent_last_name': None, 'age': None, 'full_address': None, 'contract_start_date_in_date_format': None, 'amount_paid': None, 'currency': None, 'apartment_size_sq': None}\n"
                "Decisions: [\"dict_state['respondent_first_name'] = 'איתמר'\", \"dict_state['respondent_last_name'] = 'נבון'\", \"dict_state['age'] = 33\", \"dict_state['full_address'] = 'רחוב הדקל 5'\", \"dict_state['contract_start_date_in_date_format'] = '2021-09-01'\", \"dict_state['amount_paid'] = 1800\", \"dict_state['currency'] = 'EUR'\", \"dict_state['apartment_size_sq'] = 55\"]\n\n"
                "Example 8 (EN, inline correction within the text):\n"
                "Text: 'I'm Daniel, first name Daniel, last name Cohen. I pay 900 dollars… actually, it's 950 per month. Address is 7 Maple Ave. Lease began on 03-12-2020.'\n"
                "dict_state = {'respondent_first_name': None, 'respondent_last_name': None, 'amount_paid': None, 'currency': None, 'full_address': None, 'contract_start_date_in_date_format': None}\n"
                "Decisions: [\"dict_state['respondent_first_name'] = 'Daniel'\", \"dict_state['respondent_last_name'] = 'Cohen'\", \"dict_state['amount_paid'] = 950\", \"dict_state['currency'] = 'USD'\", \"dict_state['full_address'] = '7 Maple Ave'\", \"dict_state['contract_start_date_in_date_format'] = '2020-03-12'\"]\n\n"
                "Example 9 (HE, מזהה כמחרוזת + הרבה שדות):\n"
                "Text: 'שמי נועה אברמוב. ת.ז. 123456789. אני בת 27. הכתובת: רחוב הגבעה 3. נכנסתי לדירה ב-10.02.2022. אני משלמת 4200 ש\"ח על 68 מטר.'\n"
                "dict_state = {'respondent_first_name': None, 'respondent_last_name': None, 'id': None, 'age': None, 'full_address': None, 'contract_start_date_in_date_format': None, 'amount_paid': None, 'currency': None, 'apartment_size_sq': None}\n"
                "Dec: [\"dict_state['respondent_first_name'] = 'נועה'\", \"dict_state['respondent_last_name'] = 'אברמוב'\", \"dict_state['id'] = '123456789'\", \"dict_state['age'] = 27\", \"dict_state['full_address'] = 'רחוב הגבעה 3'\", \"dict_state['contract_start_date_in_date_format'] = '2022-02-10'\", \"dict_state['amount_paid'] = 4200\", \"dict_state['currency'] = 'ש\\\"ח'\", \"dict_state['apartment_size_sq'] = 68\"]\n\n"
                "Example 10 (EN, complex dates + birth_date + many fields):\n"
                "Text: 'I am Michael Levi. Birth date is 05 Feb 1988; I am 37. I live at 24 River Road. I moved in on 15th August 2024. I pay 1500 USD for an apartment of about 75 square meters.'\n"
                "dict_state = {'respondent_first_name': None, 'respondent_last_name': None, 'birth_date': None, 'age': None, 'full_address': None, 'contract_start_date_in_date_format': None, 'amount_paid': None, 'currency': None, 'apartment_size_sq': None}\n"
                "Decisions: [\"dict_state['respondent_first_name'] = 'Michael'\", \"dict_state['respondent_last_name'] = 'Levi'\", \"dict_state['birth_date'] = '1988-02-05'\", \"dict_state['age'] = 37\", \"dict_state['full_address'] = '24 River Road'\", \"dict_state['contract_start_date_in_date_format'] = '2024-08-15'\", \"dict_state['amount_paid'] = 1500\", \"dict_state['currency'] = 'USD'\", \"dict_state['apartment_size_sq'] = 75\"]\n\n"
                "Now process the input."
            ),
            command=(
                f"Transcript so far: {full_text}\n"
                f"Current dict_state = {dict_state}\n"
                "YOUR DECISIONS:"
            )
        )

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
