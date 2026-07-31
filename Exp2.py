#Exp2
import textwrap
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)

# =========================================================
# A. SENTIMENT ANALYSIS (DistilBERT Direct Model)
# =========================================================
sent_model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)

sent_text = "The Generative AI workshop was extremely informative and useful."

inputs = sent_tokenizer(sent_text, return_tensors="pt")
with torch.no_grad():
    logits = sent_model(**inputs).logits

probs = torch.softmax(logits, dim=-1)[0]
pred_idx = torch.argmax(probs).item()
label = sent_model.config.id2label[pred_idx]
score = probs[pred_idx].item()

# =========================================================
# B. ZERO-SHOT DOCUMENT CLASSIFICATION (BART NLI Model)
# =========================================================
zs_model_name = "facebook/bart-large-mnli"
zs_tokenizer = AutoTokenizer.from_pretrained(zs_model_name)
zs_model = AutoModelForSequenceClassification.from_pretrained(zs_model_name)

document = "Artificial Intelligence and Machine Learning are transforming industries through automation and intelligent decision-making."
labels = ["Technology", "Sports", "Politics", "Entertainment"]

# NLI Hypothesis formulation for zero-shot classification
hypothesis_template = "This text is about {}."
pairs = [(document, hypothesis_template.format(label)) for label in labels]

zs_inputs = zs_tokenizer([p[0] for p in pairs], [p[1] for p in pairs], return_tensors="pt", padding=True)
with torch.no_grad():
    zs_logits = zs_model(**zs_inputs).logits

# Extract entailment probabilities (index 2 for MNLI models)
entail_logits = zs_logits[:, 2]
scores = torch.softmax(entail_logits, dim=-1).tolist()
results = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)

# =========================================================
# COMPACT SINGLE-SCREEN DISPLAY
# =========================================================
print("=" * 60)
print(" PART A: SENTIMENT ANALYSIS OUTPUT")
print("=" * 60)
print(f"Text      : '{sent_text}'")
print(f"Sentiment : {label} (Confidence: {score:.4f})")

print("\n" + "=" * 60)
print(" PART B: DOCUMENT CLASSIFICATION OUTPUT")
print("=" * 60)
print("Top Predicted Label:")
top_label, top_score = results[0]
print(f" -> {top_label}: {top_score * 100:.2f}%")

print("\nAll Scores Breakdown:")
for lbl, scr in results:
    print(f" - {lbl:<15}: {scr:.4f}")
print("=" * 60)
