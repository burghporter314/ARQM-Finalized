import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForSequenceClassification, AutoTokenizer, \
    AutoModel
import torch
from torch.quantization import quantize_dynamic
from torch.nn.functional import cosine_similarity
import pandas as pd

class GenerativeIdentificationModel:

    def __init__(self, content = None, sentences = None):
        """
        Initializes a RequirementIdentifier object

        Parameters:
        - content (string): The content of the requirements document
        - sentences (list): A list of the sentences within the requirements document
        """

        self.content = content
        self.sentences = sentences

        # Initialize the pre-trained models
        self.sentence_similarity_model = BertForSequenceClassification.from_pretrained('app/models/sentence_similarity/bert-base-requirements-identification-sentence-similarity-generative-ai')
        self.requirement_identification_model = BertForSequenceClassification.from_pretrained('app/models/requirement_identification/bert-base-requirements-identification-generative-ai')

        self.requirement_identification_model_few_shot = AutoModel.from_pretrained('app/models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese')
        # self.requirement_identification_model = quantize_dynamic(
        #     self.requirement_identification_model, {torch.nn.Linear}, dtype=torch.qint8
        # )

        self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", use_fast=True)

        self.tokenizer_few_shot = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        df_pure = pd.read_csv('app/datasets/requirement_identification/PURE_test.csv')
        df_custom = pd.read_excel(
            'app/datasets/requirement_identification/dataset.xlsx')

        df = pd.concat([df_pure, df_custom], ignore_index=True)
        df['classification'] = df['classification'].map({'T': 1, 'F': 0})
        df['text'] = df['text'].astype(str)

        self.df = df

    def getSentenceSimilarity(self, prev_sentence="", next_sentence=""):

        combined_message = (
          f"Prev Sentence: {prev_sentence}\n"
          f"Next Sentence: {next_sentence}"
        )

        inputs = self.tokenizer(
            prev_sentence,
            next_sentence,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        outputs = self.sentence_similarity_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()


        return pred

    def getIfRequirement(self, texts, batch_size=64):
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
            )

            with torch.no_grad():
                outputs = self.requirement_identification_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                # Filter based on confidence threshold
                for pred, prob in zip(preds.tolist(), max_probs.tolist()):
                    if prob > 0.95:
                        all_preds.append(pred)
                    else:
                        all_preds.append(None)  # or any placeholder for low confidence


        return all_preds

    def getIfRequirementFewShot(self, texts, threshold=0.975, batch_size=32):

        ref_df = self.df.groupby('classification', group_keys=False).apply(
            lambda x: x.sample(n=300, random_state=42)).reset_index(drop=True)

        reference_texts = ref_df['text'].tolist()
        reference_labels = ref_df['classification'].tolist()

        with torch.no_grad():
            ref_inputs = self.tokenizer_few_shot(reference_texts, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=32)
            ref_embs = self.requirement_identification_model_few_shot(**ref_inputs).last_hidden_state[:, 0, :]

        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                inputs = self.tokenizer_few_shot(batch_texts, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=32)
                embs = self.requirement_identification_model_few_shot(**inputs).last_hidden_state[:, 0, :]

                sims = cosine_similarity(embs.unsqueeze(1), ref_embs.unsqueeze(0), dim=-1)
                max_sims, max_indices = torch.max(sims, dim=1)

                for sim, idx in zip(max_sims, max_indices):
                    if sim.item() > threshold and reference_labels[idx.item()] == 1:
                        predictions.append(1)
                    else:
                        predictions.append(0)

        return predictions