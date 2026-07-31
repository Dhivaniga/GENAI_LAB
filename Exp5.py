#Exp 5

import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------
# Load Flan-T5 model directly (No pipeline task KeyError)
# ---------------------------------------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_response(prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------
content_prompt = "As an AI teacher, explain Generative AI to 1st year engineers simply with 2 real-world applications in 3 sentences."
reasoning_prompt = "120 total students attended a workshop, 85 completed it. How many did not complete it? Show simple steps and answer."
email_prompt = "Write a short email for GenAI lab: Friday 10 AM, AI Lab 2. Bring laptops and register Hugging Face account."

meeting_notes = "Arun: prepare dataset by Monday. Priya: test chatbot by Wednesday. Rahul: prep demo presentation."
action_prompt = f"Extract action items as numbered list (Person, Task, Deadline) from: {meeting_notes}"

# ---------------------------------------------------------
# Compact Single-Screen Output Display
# ---------------------------------------------------------
print("=" * 60)
print(" PROMPT ENGINEERING APPLICATION")
print("=" * 60)

print("[1. CONTENT GENERATION]")
print(textwrap.fill(generate_response(content_prompt), width=60))

print("\n[2. REASONING TASK]")
print(textwrap.fill(generate_response(reasoning_prompt), width=60))

print("\n[3. EMAIL GENERATION]")
print(textwrap.fill(generate_response(email_prompt), width=60))

print("\n[4. ACTION ITEM EXTRACTION]")
print(textwrap.fill(generate_response(action_prompt), width=60))
print("=" * 60)
