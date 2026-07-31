#exp1
import textwrap
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------------------------------------
# Load pretrained GPT-2 model and tokenizer
# ---------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set pad_token_id to eos_token_id to prevent generation warnings
tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = "Artificial Intelligence is"

# Tokenize input
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs,
    max_length=60,          # Kept concise so output stays compact
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------
# Compact Single-Screen Display Output
# ---------------------------------------------------------
print("=" * 60)
print(" GPT-2 TEXT GENERATION OUTPUT")
print("=" * 60)
print(f"Prompt : '{prompt}'\n")
print("Generated Continuation:")
print(textwrap.fill(generated_text, width=60))
print("=" * 60)
