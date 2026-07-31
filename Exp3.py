#exp3
import torch
import textwrap
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForQuestionAnswering
)

# =========================================================
# A. SUMMARIZATION (BART Direct Inference)
# =========================================================
sum_model_name = "facebook/bart-large-cnn"
sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)

text = (
    "Artificial Intelligence is transforming many industries by enabling machines "
    "to perform tasks that normally require human intelligence. It is widely used in "
    "healthcare, education, manufacturing, finance, transportation, and cybersecurity. "
    "AI systems can analyze large amounts of data, identify patterns, make predictions, "
    "and support intelligent decision-making. Generative AI is a branch of Artificial "
    "Intelligence that can create new content such as text, images, audio, video, and computer programs."
)

inputs = sum_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = sum_model.generate(
    inputs["input_ids"], 
    max_length=60, 
    min_length=20, 
    do_sample=False
)
summary_text = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# =========================================================
# B. QUESTION ANSWERING (DistilBERT Direct Inference)
# =========================================================
qa_model_name = "distilbert-base-cased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

context = (
    "Generative Artificial Intelligence is a type of Artificial Intelligence that can "
    "create new content such as text, images, audio, video, and computer programs. "
    "Large Language Models are commonly used for text generation, summarization, "
    "translation, and question answering."
)
question = "What type of content can Generative AI create?"

qa_inputs = qa_tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = qa_model(**qa_inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer_tokens = qa_inputs["input_ids"][0][start_idx:end_idx]
answer_text = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

start_score = torch.softmax(outputs.start_logits, dim=-1)[0][start_idx]
end_score = torch.softmax(outputs.end_logits, dim=-1)[0][end_idx - 1]
confidence_score = float((start_score * end_score).item())

# =========================================================
# COMPACT SCREEN OUTPUT (Single Screen View)
# =========================================================
print("=" * 62)
print(" PART A: SUMMARIZATION OUTPUT")
print("=" * 62)
print(textwrap.fill(summary_text, width=62))

print("\n" + "=" * 62)
print(" PART B: QUESTION ANSWERING OUTPUT")
print("=" * 62)
print(f"Question   : {question}")
print(f"Answer     : {answer_text}")
print(f"Confidence : {confidence_score:.3f}")
print("=" * 62)
