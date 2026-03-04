import os
import json
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from torch.quantization import quantize_dynamic


class GenerativeIdentificationModel:

    def __init__(self, content=None, sentences=None):
        """
        Initializes a RequirementIdentifier object

        Parameters:
        - content (string): The content of the requirements document
        - sentences (list): A list of the sentences within the requirements document
        """
        self.content = content
        self.sentences = sentences

        # Base directory for relative paths (works in Docker)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Models
        self.sentence_similarity_model = BertForSequenceClassification.from_pretrained(
            os.path.join(BASE_DIR, "../../models/sentence_similarity/bert-base-requirements-identification-sentence-similarity-generative-ai")
        )
        self.requirement_identification_model = BertForSequenceClassification.from_pretrained(
            os.path.join(BASE_DIR, "../../models/requirement_identification/bert-base-requirements-identification-generative-ai")
        )
        self.requirement_identification_model_few_shot = AutoModel.from_pretrained(
            os.path.join(BASE_DIR, "../../models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese")
        )

        self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", use_fast=True)
        self.tokenizer_few_shot = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Dataset paths
        df_pure_path = os.path.join(BASE_DIR, "../../datasets/requirement_identification/PURE_test.csv")
        df_custom_path = os.path.join(BASE_DIR, "../../datasets/requirement_identification/dataset.xlsx")

        df_pure = pd.read_csv(df_pure_path)
        df_custom = pd.read_excel(df_custom_path)

        # Normalize column names
        df_pure.columns = df_pure.columns.str.strip().str.lower()
        df_custom.columns = df_custom.columns.str.strip().str.lower()

        # Combine
        df = pd.concat([df_pure, df_custom], ignore_index=True)
        df['classification'] = df['classification'].map({'T': 1, 'F': 0})
        df['text'] = df['text'].astype(str)

        self.df = df

    # ---------------------------
    # Sentence similarity
    # ---------------------------
    def getSentenceSimilarity(self, prev_sentence="", next_sentence=""):
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

    # ---------------------------
    # Batch classification
    # ---------------------------
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

                for pred, prob in zip(preds.tolist(), max_probs.tolist()):
                    all_preds.append(pred if prob > 0.95 else None)

        return all_preds

    # ---------------------------
    # Few-shot requirement identification
    # ---------------------------
    def getIfRequirementFewShot(self, texts, threshold=0.975, batch_size=32):

        print("DEBUG: Columns in df:", self.df.columns.tolist())
        print("DEBUG: Head of df:\n", self.df.head())
        print("DEBUG: Unique values in classification column:", self.df['classification'].unique())
        print("DEBUG: Null values in classification column:", self.df['classification'].isnull().sum())
        print("DEBUG: Count per classification value:\n", self.df['classification'].value_counts())
        ref_df_list = []

        for cls, group in self.df.groupby('classification'):
            # Sample 300 rows or all if fewer than 300
            n = min(len(group), 300)
            ref_df_list.append(group.sample(n=n, random_state=42))

        ref_df = pd.concat(ref_df_list, ignore_index=True)

        # Now extract texts and labels safely
        reference_texts = ref_df['text'].tolist()
        reference_labels = ref_df['classification'].tolist()

        with torch.no_grad():
            ref_inputs = self.tokenizer_few_shot(reference_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
            ref_embs = self.requirement_identification_model_few_shot(**ref_inputs).last_hidden_state[:, 0, :]

        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                inputs = self.tokenizer_few_shot(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
                embs = self.requirement_identification_model_few_shot(**inputs).last_hidden_state[:, 0, :]

                sims = cosine_similarity(embs.unsqueeze(1), ref_embs.unsqueeze(0), dim=-1)
                max_sims, max_indices = torch.max(sims, dim=1)

                for sim, idx in zip(max_sims, max_indices):
                    predictions.append(1 if sim.item() > threshold and reference_labels[idx.item()] == 1 else 0)

        return predictions