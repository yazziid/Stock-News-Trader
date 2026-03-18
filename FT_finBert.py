import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", local_files_only=True)

def get_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    negative = probs[0][0].item()
    neutral = probs[0][1].item()
    positive = probs[0][2].item()

    score = positive - negative

    return score