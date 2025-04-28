import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
from sklearn.model_selection import KFold
from transformers import EarlyStoppingCallback, pipeline
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
label_columns = ["result_binary_ambiguity", "result_binary_feasibility", "result_binary_singularity",
                 "result_binary_verifiability"]

import random
import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
import re
import shap
import time

import nlpaug.augmenter.word as naw


max_length = 32

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def train(load_saved = False, save_result = False, num_synthetic = 0, rebalance_classes = False, truncate_beginning_words=0, metric_for_best_model="eval_loss", model_name="bert-base-uncased", max_length=32, k_folds=5, train_batch_size = 4, eval_batch_size = 4, epochs=10):

    # Load dataset
    file_path = "app/datasets/requirement_quality/dataset.xlsx"  # Update this path
    df = pd.read_excel(file_path)

    # Convert 'T'/'F' to binary labels (1 and 0)
    df[label_columns] = df[label_columns].replace({'T': 1, 'F': 0})

    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='substitute', aug_p=0.15)

    # Rebalance dataset (oversample the minority class)
    if rebalance_classes:
        for label in label_columns:
            minority_class = df[df[label] == df[label].value_counts().idxmin()]
            majority_class = df[df[label] == df[label].value_counts().idxmax()]
            oversampled_minority = minority_class.sample(len(majority_class), replace=True)
            df = pd.concat([majority_class, oversampled_minority], axis=0)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=35).reset_index(drop=True)

    # Convert text column to string
    df["requirement"] = df["requirement"].astype(str)
    df = df.drop("result", axis=1)

    if(num_synthetic > 0):

        def is_valid_word(word):
            """Check if the word is a valid word in WordNet."""
            return wordnet.synsets(word)

        def get_contextual_synonyms(word, sentence):
            # Use the Lesk algorithm to find the word's sense in context
            context_synset = lesk(nltk.word_tokenize(sentence), word)

            if context_synset:
                # Get synonyms from the identified sense
                synonyms = context_synset.lemmas()
                # Filter valid single-word synonyms
                valid_synonyms = [synonym.name() for synonym in synonyms if synonym.name() != word and is_valid_word(
                    synonym.name()) and '_' not in synonym.name()]
                return valid_synonyms
            else:
                return []

        def replace_random_words_with_synonyms(text, num_replacements=3):
            words = nltk.word_tokenize(text)  # Tokenize the sentence into words
            words = [word for word in words if word.isalpha()]  # Remove punctuation

            random_words = random.sample(words, min(num_replacements, len(words)))  # Randomly select words to replace

            for word in random_words:
                # Get contextual synonyms for the word
                synonyms = get_contextual_synonyms(word, text)

                if synonyms:  # If synonyms exist
                    # Pick a random synonym from the list of valid synonyms
                    synonym = random.choice(synonyms)

                    # Replace the word with the synonym using word boundaries (\b)
                    text = re.sub(r'\b' + re.escape(word) + r'\b', synonym, text, count=1)  # Replace word only once

            return text

        # Example DataFrame (replace this with your actual df)
        df_sampled = df.sample(num_synthetic, replace=True)

        for i, row in df_sampled.iterrows():
            original_text = row["requirement"]
            print("Original:", original_text)

            # Replace three random words with their synonyms
            modified_text = aug.augment(original_text)[0]

            print("Modified:", modified_text)

            # Update the requirement column with the modified text
            df_sampled.at[i, "requirement"] = modified_text

        # Concatenate the original DataFrame and the modified DataFrame
        df = pd.concat([df, df_sampled], ignore_index=True)

    # Load tokenizer
    if load_saved:
        tokenizer = AutoTokenizer.from_pretrained("./bert-requirement-classifier")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df["requirement"] = df["requirement"].apply(
        lambda x: " ".join(x.split()[truncate_beginning_words:])
    )

    # Convert dataset to Hugging Face Dataset format
    dataset = Dataset.from_pandas(df)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["requirement"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    # Correctly map labels
    def label_map(example):
        return {"labels": torch.tensor([example[col] for col in label_columns], dtype=torch.float)}

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Map labels
    tokenized_dataset = tokenized_dataset.map(label_map, batched=False)

    # Split dataset into training and validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.3)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Data collator (for dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(pred):
        logits = pred.predictions  # Extract logits from tuple
        if isinstance(logits, tuple):
            logits = logits[0]  # Take the first element if logits is a tuple

        preds = (np.array(logits) > 0.5).astype(int)  # Convert to NumPy array before thresholding

        labels = pred.label_ids

        # Compute overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=1)
        accuracy = accuracy_score(labels, preds)

        # Compute per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(labels, preds,
                                                                                                 average=None,
                                                                                                 zero_division=1)

        # Store metrics
        metrics = {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }

        # Add per-class metrics
        for i, label_name in enumerate(label_columns):
            metrics[f"precision_{label_name}"] = precision_per_class[i]
            metrics[f"recall_{label_name}"] = recall_per_class[i]
            metrics[f"f1_{label_name}"] = f1_per_class[i]

        return metrics

    if load_saved:
        tokenizer = AutoTokenizer.from_pretrained("./bert-requirement-classifier")
        model = BertForSequenceClassification.from_pretrained("./bert-requirement-classifier", output_attentions=True)
    else:
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
            num_labels=len(label_columns),
            problem_type = "multi_label_classification",
            output_attentions=True
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bert-multi-label-classification-results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=1e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.05,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model=metric_for_best_model
    )

    # Define K-Fold Cross Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results = []

    # K-Fold Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(tokenized_dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split dataset into train and eval
        train_dataset = tokenized_dataset.select(train_idx)
        eval_dataset = tokenized_dataset.select(val_idx)

        # Trainer instance for each fold
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train model
        trainer.train()

        # Evaluate model
        eval_results = trainer.evaluate()
        fold_results.append(eval_results)

        print(f"Results for Fold {fold + 1}: {eval_results}")

    # Save final model
    if save_result:
        model.save_pretrained("./bert-requirement-classifier")
        tokenizer.save_pretrained("./bert-requirement-classifier")

def test(num_tests=10):

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.float16
    )

    model = BertForSequenceClassification.from_pretrained("./bert-requirement-classifier", output_attentions=True)

    tokenizer = AutoTokenizer.from_pretrained('./bert-requirement-classifier')

    def predict_function(texts):
        """SHAP-compatible prediction function for multi-label classification"""
        inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=32)
        # inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # Get model logits
        return torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities

    def get_top_ngrams(words, shap_scores, n=2, top_k=3):
        """Extracts top n-grams with highest SHAP importance."""
        if len(words) < n:
            return []  # Skip if text is too short for n-grams
        ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        ngram_scores = [sum(shap_scores[i:i + n]) for i in range(len(shap_scores) - n + 1)]
        return sorted(zip(ngrams, ngram_scores), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    # Create a SHAP explainer using the correct model and tokenizer
    explainer = shap.Explainer(predict_function, tokenizer, batch_size=32, max_evals=150)

    def predict_requirement(texts, num_predictions=10, threshold=0.5, max_n=5):
        inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        # inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors only

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()

        # Only explain for those that meet conditions
        explain_indices = []
        for i in range(num_predictions):
            should_explain = False
            p = probs[i]
            if p[0] > 0.5 or any(p[j] < 0.5 for j in [1, 2, 3]):
                explain_indices.append(i)

        if explain_indices:
            explain_texts = [texts[i] for i in explain_indices]
            shap_values = explainer(explain_texts)
        else:
            shap_values = None

        for i in range(num_predictions):
            print(f"\nRequirement {i+1}: {texts[i]}")
            print(f"\nPrediction {i + 1}:")
            predictions = (probs[i] > threshold).astype(int)
            result = {label_columns[j]: predictions[j] for j in range(len(label_columns))}
            print("Predicted labels:", result)

            if i in explain_indices:
                sv_idx = explain_indices.index(i)
                words = shap_values.data[sv_idx]
                for j, label in enumerate(label_columns):
                    p = probs[i][j]
                    if (j == 0 and p > 0.5) or (j in [1, 2, 3] and p < 0.5):
                        shap_scores = shap_values.values[sv_idx][:, j]
                        for n in range(2, max_n + 1):
                            print(f"\nTop {n}-grams:")
                            for ngram, score in get_top_ngrams(words, shap_scores, n=n):
                                print(f"{n}-gram: '{ngram}' - Impact: {abs(score):.4f}")






    # Example usage
    df = pd.read_excel('app/datasets/requirement_quality/dataset.xlsx')
    texts = df['requirement'].sample(num_tests, replace=True).tolist()
    predict_requirement(texts)


start_time = time.time()  # Start timing

train(load_saved = False, save_result = True, model_name='distilbert-base-uncased', num_synthetic=2000, rebalance_classes=False, k_folds = 10, train_batch_size = 16, eval_batch_size=16, epochs = 5)
test(num_tests=500)

end_time = time.time()  # End timing
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")