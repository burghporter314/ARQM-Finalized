from app.util.model_impl.wrappers.bert_wrapper import BertWrapper

from transformers import T5ForConditionalGeneration
import torch
from torch.quantization import quantize_dynamic


class GenerativeQualityModel:

    def __init__(self, content=None, sentences=None):
        """
        Initializes a GenerativeQualityModel object
        """

        self.bert_wrapper = BertWrapper(violation_threshold=.8, display_ngram_summary=False)

    def getOverallQuality(self, texts, batch_size=32):
        self.bert_wrapper.predict_requirement_v2(texts)