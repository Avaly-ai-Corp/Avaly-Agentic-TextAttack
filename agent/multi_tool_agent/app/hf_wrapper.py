from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import ModelWrapper
import torch
import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class MyHFWrapper(ModelWrapper):

    def __init__(self, name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=os.getenv("HF_ALLOW_CODE_EVAL", "false").lower()
            == "true",
        )
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name,
            trust_remote_code=os.getenv("HF_ALLOW_CODE_EVAL", "false").lower()
            == "true",
        )
        self.model.to(self.device).eval()

    def __call__(self, text_list):
        encoded = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = encoded.to(self.model.device)
        with torch.no_grad():
            return self.model(**encoded).logits


model = MyHFWrapper(os.getenv("MODEL_NAME"))
