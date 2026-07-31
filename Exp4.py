#exp4
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# Load DialoGPT Model and Tokenizer
# ---------------------------------------------------------
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token ID explicitly
tokenizer.pad_token_id = tokenizer.eos_token_id

# Sample turns to demonstrate conversation flow for report screenshot
simulated_inputs = [
    "Hello! How are you doing today?",
    "Can you tell me what you like to do?",
    "exit"
]

print("=" * 60)
print(" DIALOGPT CHATBOT DEMO")
print("=" * 60)

chat_history_ids = None
input_idx = 0

while True:
    # Simulates input for a clean output; replace with input("You: ") for manual chat
    user_input = simulated_inputs[input_idx]
    input_idx += 1
    
    print(f"\nYou: {user_input}")
    
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye! Have a nice day.")
        break

    # Encode user input with end-of-sequence token
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, 
        return_tensors="pt"
    )

    # Append to chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    # Generate response (keep max_length sensible for multi-turn history)
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 50,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    
    wrapped_resp = textwrap.fill(response, width=50, subsequent_indent="         ")
    print(f"Chatbot: {wrapped_resp}")

print("=" * 60)
