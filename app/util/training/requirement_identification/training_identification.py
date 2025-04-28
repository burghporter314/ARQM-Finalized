import pandas as pd
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, BertForSequenceClassification, EarlyStoppingCallback, TrainingArguments, Trainer
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from torch.nn.functional import pairwise_distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity

#
# from sklearn.neural_network import MLPClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l1, l2
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
#
#
#
# def load_and_preprocess():
#     # Load dataset
#     df = pd.read_excel('../../data/requirements_identification_generative_ai/requirement_identification/dataset.xlsx')
#
#     # Convert classification to binary (T -> 1, F -> 0)
#     df['classification'] = df['classification'].map({'T': 1, 'F': 0})
#
#     # Preprocessing
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
#     X = vectorizer.fit_transform(df['text']).toarray()
#     y = df['classification'].values
#
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)
#
#     return X_train, X_test, y_train, y_test, vectorizer
#
# def load_and_preprocess_cnn_rnn(vocab_size=5000, max_len=500):
#
#     df = pd.read_excel('../../data/requirements_identification_generative_ai/requirement_identification/dataset.xlsx')
#
#     df['classification'] = df['classification'].map({'T': 1, 'F': 0})
#
#     X = df['text'].values
#     y = df['classification'].values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)
#
#     tokenizer = Tokenizer(num_words=vocab_size)
#     tokenizer.fit_on_texts(X_train)
#     X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
#     X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)
#
#     return X_train, X_test, y_train, y_test, tokenizer
#
# def get_linear_regression_model():
#     return LinearRegression()
#
# def get_logistic_regression_model():
#     return LogisticRegression()
#
# def get_naive_bayes_model():
#     return MultinomialNB()
#
# def get_svm_model():
#     return SVC(kernel='linear', probability=True)
#
# def get_random_forest_model():
#     return RandomForestClassifier(n_estimators=250)
#
# def get_knn_model():
#     return KNeighborsClassifier(n_neighbors=20)
#
# def get_ann_model(input_dim):
#     model = Sequential()
#
#     model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
# def get_cnn_model(vocab_size=5000, max_len=500):
#     model = Sequential()
#
#     model.add(Embedding(vocab_size, 128, input_length=max_len))
#
#     model.add(Conv1D(128, 5, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.2))
#
#     model.add(Conv1D(64, 3, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.2))
#
#     model.add(Conv1D(32, 3, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.2))
#
#     model.add(Flatten())
#
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(1, activation='sigmoid'))
#
#     return model
#
# def get_rnn_model(vocab_size=5000, max_len=500):
#
#     model = Sequential()
#
#     model.add(Embedding(vocab_size, 128, input_length=max_len))
#
#     model.add(LSTM(128, return_sequences=True))
#     model.add(Dropout(0.2))
#
#     model.add(LSTM(64, return_sequences=True))
#     model.add(Dropout(0.2))
#
#     model.add(LSTM(32))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(1, activation='sigmoid'))
#
#     return model
#
# def train_linear_regression():
#
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
#
#     # Fit the linear regression model
#     model = get_linear_regression_model()
#     model.fit(X_train, y_train)
#
#     # Predict using linear regression. Since it returns probabilities, convert to 0 and 1.
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)
#
#     print(classification_report(y_test, y_pred))
#
# def train_logistic_regression():
#
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
#
#     model = get_logistic_regression_model()
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#
#     print(classification_report(y_test, y_pred))
#
# def train_naive_bayes():
#
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
#
#     model = get_naive_bayes_model()
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))
#
# def train_svm():
#
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
#
#     model = get_svm_model()
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)
#
#     print(classification_report(y_test, y_pred))
#
# def train_random_forest():
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
#
#     model = get_random_forest_model()
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))
#
# def train_knn():
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
#
#     model = get_knn_model()
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))
#
# def train_ann():
#
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
#
#     input_dim = X_train.shape[1]
#
#     model = get_ann_model(input_dim)
#
#     model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)
#
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)
#
#     print(classification_report(y_test, y_pred))
#
# def train_cnn(vocab_size=5000, max_len=500):
#
#     X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_cnn_rnn()
#
#     model = get_cnn_model()
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=60, batch_size=4, verbose=1)
#
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)
#
#     print(classification_report(y_test, y_pred))
#
# def train_rnn(vocab_size=5000, max_len=500):
#     X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_cnn_rnn(vocab_size, max_len)
#
#     model = get_rnn_model()
#
#     model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
#
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     print(classification_report(y_test, y_pred))
#
#
# def train_ensemble(vocab_size=5000, max_len=500):
#     # Preprocess data for traditional models and deep learning models
#     X_train_traditional, X_test_traditional, y_train, y_test, vectorizer = load_and_preprocess()
#     X_train_cnn_rnn, X_test_cnn_rnn, _, _, tokenizer = load_and_preprocess_cnn_rnn(vocab_size, max_len)
#
#     # Define models
#     models_traditional = {
#         "linear_regression": get_linear_regression_model(),
#         "logistic_regression": get_logistic_regression_model(),
#         "naive_bayes": get_naive_bayes_model(),
#         "support_vector_machine": get_svm_model(),
#         "random_forest": get_random_forest_model()
#     }
#     models_dl = {
#         "cnn": get_cnn_model(),
#         "rnn": get_rnn_model()
#     }
#
#     # Train traditional models and collect predictions
#     train_predictions = {}
#     test_predictions = {}
#
#     for name, model in models_traditional.items():
#         model.fit(X_train_traditional, y_train)
#         if hasattr(model, "predict_proba"):
#             train_predictions[name] = model.predict_proba(X_train_traditional)[:, 1]
#             test_predictions[name] = model.predict_proba(X_test_traditional)[:, 1]
#         else:
#             train_predictions[name] = np.clip(model.predict(X_train_traditional), 0, 1)
#             test_predictions[name] = np.clip(model.predict(X_test_traditional), 0, 1)
#
#         print(name)
#         print(classification_report(y_test, (test_predictions[name] > 0.5).astype(int)))
#
#
#     # Train deep learning models and collect predictions
#     for name, model in models_dl.items():
#         if name == 'cnn':
#             model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#             model.fit(X_train_cnn_rnn, y_train, epochs=60, batch_size=8, verbose=1)
#         else:
#             model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#             model.fit(X_train_cnn_rnn, y_train, epochs=8, batch_size=4, verbose=1)
#
#         train_predictions[name] = model.predict(X_train_cnn_rnn).flatten()
#         test_predictions[name] = model.predict(X_test_cnn_rnn).flatten()
#
#         print(name)
#         print(classification_report(y_test, (test_predictions[name] > 0.5).astype(int)))
#
#     # Stack predictions for training and testing
#     stacked_train = np.stack([train_predictions[name] for name in train_predictions.keys()], axis=1)
#     stacked_test = np.stack([test_predictions[name] for name in test_predictions.keys()], axis=1)
#
#     # --- Average Ensemble ---
#     avg_probs_test = np.mean(stacked_test, axis=1)
#     avg_ensemble_preds = (avg_probs_test > 0.5).astype(int)
#     print("Average Ensemble")
#     print(classification_report(y_test, avg_ensemble_preds))
#
#     # --- Majority Vote Ensemble ---
#     binary_preds_test = (stacked_test > 0.5).astype(int)
#     majority_vote_preds = (np.sum(binary_preds_test, axis=1) > (binary_preds_test.shape[1] / 2)).astype(int)
#     print("Majority Vote Ensemble")
#     print(classification_report(y_test, majority_vote_preds))
#
#     # Train and evaluate Logistic Regression meta-model
#     meta_model_lr = LogisticRegression()
#     meta_model_lr.fit(stacked_train, y_train)
#     ensemble_preds_lr = meta_model_lr.predict(stacked_test)
#     print("Logistic Regression Meta-Model")
#     print(classification_report(y_test, ensemble_preds_lr))
#
#     # Train and evaluate Gradient Boosting meta-model
#     meta_model_gb = GradientBoostingClassifier()
#     meta_model_gb.fit(stacked_train, y_train)
#     ensemble_preds_gb = meta_model_gb.predict(stacked_test)
#     print("Gradient Boosting Meta-Model")
#     print(classification_report(y_test, ensemble_preds_gb))

def train_bert():
    df = pd.read_excel('../../data/requirements_identification_generative_ai/requirement_identification/dataset.xlsx')

    print(df['classification'].value_counts())

    df['classification'] = df['classification'].map({'T': 1, 'F': 0})

    # Rebalance

    minority_class = df[df['classification'] == df['classification'].value_counts().idxmin()]
    majority_class = df[df['classification'] == df['classification'].value_counts().idxmax()]

    # Oversample the minority class
    oversampled_minority = minority_class.sample(len(majority_class), replace=True)

    # Combine to get the balanced dataset
    df = pd.concat([majority_class, oversampled_minority], axis=0).sample(frac=1).reset_index(drop=True)

    df['text'] = df['text'].astype(str)

    print(df['classification'].value_counts())

    # Randomize the entries
    df = df.sample(len(df) * 5, random_state=35, replace=True)

    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
    #                                                       hidden_dropout_prob=0.2,
    #                                                       attention_probs_dropout_prob=0.2,
    #                                                       num_labels=2)

    model = BertForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",
                                                          hidden_dropout_prob=0.3,
                                                          attention_probs_dropout_prob=0.3,
                                                          num_labels=2)

    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128,
            padding="max_length"
        )


    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['classification']}, batched=True)

    split_dataset = tokenized_dataset.train_test_split(test_size=0.3)

    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        accuracy = accuracy_score(labels, preds)

        metrics = {
            'accuracy': accuracy
        }

        for i in range(len(precision)):
            metrics[f'precision_class_{i}'] = precision[i]
            metrics[f'recall_class_{i}'] = recall[i]
            metrics[f'f1_class_{i}'] = f1[i]

        return metrics

    training_args = TrainingArguments(
        output_dir='../bert-binary-classification-results',
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.05,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss"
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=6)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Define K-Fold Cross Validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results = []

    # K-Fold Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(tokenized_dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split dataset into train and eval based on indices
        train_dataset = tokenized_dataset.select(train_idx)
        eval_dataset = tokenized_dataset.select(val_idx)

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        fold_results.append(eval_results)

        print(f"Results for Fold {fold + 1}: {eval_results}")

    # Print summary of results for all folds
    for i, result in enumerate(fold_results):
        print(f"Fold {i + 1} - Loss: {result['eval_loss']}")


    test_predictions = trainer.predict(eval_dataset)
    test_labels = test_predictions.label_ids
    test_preds = np.argmax(test_predictions.predictions, axis=1)

    print("Classification Report for Test Data")
    print(classification_report(test_labels, test_preds, target_names=["Class 0 (F)", "Class 1 (T)"]))

    model.save_pretrained("../models/bert-base-requirements-identification-generative-ai")



import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import random

def train_bert_few_shot():
    # Load and preprocess
    df = pd.read_excel('../../data/requirements_identification_generative_ai/requirement_identification/dataset.xlsx')
    df['classification'] = df['classification'].map({'T': 1, 'F': 0})
    df['text'] = df['text'].astype(str)

    # Few-shot sampling
    few_shot_df = df.groupby('classification', group_keys=False).apply(
        lambda x: x.sample(n=40, random_state=42)
    ).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    encoder = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    class TripletTextDataset(TorchDataset):
        def __init__(self, df, tokenizer, encoder, margin=0.25):
            self.df = df.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.encoder = encoder
            self.margin = margin
            self.label_to_indices = df.groupby('classification').indices

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            anchor_text = self.df.loc[idx, 'text']
            anchor_label = self.df.loc[idx, 'classification']

            # Sample positive
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = random.choice(self.label_to_indices[anchor_label])
            positive_text = self.df.loc[positive_idx, 'text']

            # Compute embeddings
            with torch.no_grad():
                anchor_emb = self.encoder(
                    **self.tokenizer(anchor_text, return_tensors="pt", padding=True, truncation=True,
                                     max_length=32)).last_hidden_state[:, 0, :]
                pos_emb = self.encoder(
                    **self.tokenizer(positive_text, return_tensors="pt", padding=True, truncation=True,
                                     max_length=32)).last_hidden_state[:, 0, :]
                pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, pos_emb).item()

            # Search for semi-hard negative
            hardest_neg = None
            hardest_dist = 0
            negative_idx = None

            for label, indices in self.label_to_indices.items():
                if label == anchor_label:
                    continue
                for neg_idx in indices:
                    neg_text = self.df.loc[neg_idx, 'text']
                    with torch.no_grad():
                        neg_emb = self.encoder(
                            **self.tokenizer(neg_text, return_tensors="pt", padding=True, truncation=True,
                                             max_length=32)).last_hidden_state[:, 0, :]
                        dist = torch.nn.functional.pairwise_distance(anchor_emb, neg_emb).item()

                    if pos_dist < dist < self.margin + pos_dist:
                        negative_idx = neg_idx  # Found semi-hard negative
                        break
                    if dist > hardest_dist:
                        hardest_neg = neg_idx
                        hardest_dist = dist

            if negative_idx is None:
                negative_idx = hardest_neg  # fallback to hardest

            negative_text = self.df.loc[negative_idx, 'text']

            # Tokenize triplet
            texts = [anchor_text, positive_text, negative_text]
            encoded = self.tokenizer(texts, truncation=True, padding="max_length", max_length=32, return_tensors="pt")

            return encoded, torch.tensor(anchor_label)

    # Training
    triplet_loader = DataLoader(TripletTextDataset(few_shot_df, tokenizer, encoder), batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-5)
    loss_fn = nn.TripletMarginLoss(margin=0.5)

    encoder.train()
    for epoch in range(25):
        total_loss = 0
        for batch, _ in triplet_loader:
            anchor = {k: v[0] for k, v in batch.items()}
            positive = {k: v[1] for k, v in batch.items()}
            negative = {k: v[2] for k, v in batch.items()}

            anchor_emb = encoder(**anchor).last_hidden_state[:, 0, :]
            pos_emb = encoder(**positive).last_hidden_state[:, 0, :]
            neg_emb = encoder(**negative).last_hidden_state[:, 0, :]

            loss = loss_fn(anchor_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} - Triplet Loss: {total_loss:.4f}")

    # Freeze encoder and generate embeddings
    def compute_cls_embeddings(model, tokenizer, texts):
        model.eval()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    all_embeddings = []
    all_labels = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(df), 32):
            batch_df = df.iloc[i:i + 32]
            texts = batch_df['text'].tolist()
            labels = batch_df['classification'].tolist()
            emb = compute_cls_embeddings(encoder, tokenizer, texts)
            all_embeddings.append(emb)
            all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = np.array(all_labels)

    # Train and evaluate classifier
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Class 0 (F)", "Class 1 (T)"]))

class SiameseDataset(TorchDataset):
    def __init__(self, df, tokenizer, num_pairs=1000):
        self.pairs = []
        self.labels = []
        self.tokenizer = tokenizer
        self.label_to_indices = df.groupby('classification').indices
        self.df = df.reset_index(drop=True)

        for _ in range(num_pairs):
            # Positive pair
            label = random.choice(list(self.label_to_indices.keys()))
            idx1, idx2 = random.sample(list(self.label_to_indices[label]), 2)
            self.pairs.append((df.loc[idx1, 'text'], df.loc[idx2, 'text']))
            self.labels.append(1)

            # Negative pair
            label1, label2 = random.sample(list(self.label_to_indices.keys()), 2)
            idx1 = random.choice(self.label_to_indices[label1])
            idx2 = random.choice(self.label_to_indices[label2])
            self.pairs.append((df.loc[idx1, 'text'], df.loc[idx2, 'text']))
            self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sent1, sent2 = self.pairs[idx]
        label = self.labels[idx]
        encoded = self.tokenizer([sent1, sent2], padding='max_length', truncation=True,
                                 max_length=32, return_tensors='pt')
        return encoded['input_ids'][0], encoded['attention_mask'][0], \
               encoded['input_ids'][1], encoded['attention_mask'][1], \
               torch.tensor(label, dtype=torch.float)

def collate_fn(batch):
    ids1, masks1, ids2, masks2, labels = zip(*batch)
    return (
        torch.stack(ids1),
        torch.stack(masks1),
        torch.stack(ids2),
        torch.stack(masks2),
        torch.stack(labels),
    )


class SiameseBERT(nn.Module):
    def __init__(self, model_name, dropout_prob=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        out1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state[:, 0, :]
        out2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state[:, 0, :]

        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

        return out1, out2

def cosine_contrastive_loss(out1, out2, labels, margin=0.5):
    cos = nn.functional.cosine_similarity(out1, out2)
    loss = labels * (1 - cos) ** 2 + (1 - labels) * torch.clamp(cos - margin, min=0) ** 2
    return loss.mean()

def train_siamese_model():
    df_pure = pd.read_csv('app/datasets/requirement_identification/PURE_test.csv')
    df_custom = pd.read_excel('app/datasets/requirement_identification/dataset.xlsx')

    df = pd.concat([df_pure, df_custom], ignore_index=True)

    df = df.sample(frac=1).reset_index(drop=True)
    df['classification'] = df['classification'].map({'T': 1, 'F': 0})
    df['text'] = df['text'].astype(str)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = SiameseBERT('distilbert-base-uncased')
    dataset = SiameseDataset(df, tokenizer, num_pairs=400)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(40):
        total_loss = 0
        for input_ids1, attn1, input_ids2, attn2, labels in loader:
            out1, out2 = model(input_ids1, attn1, input_ids2, attn2)
            loss = cosine_contrastive_loss(out1, out2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for i in range(0, len(df), 32):
            batch = df.iloc[i:i+32]
            tokens = tokenizer(batch['text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=32)
            out = model.encoder(**tokens).last_hidden_state[:, 0, :]
            embeddings.append(out)
            labels.extend(batch['classification'].tolist())

    embeddings = torch.cat(embeddings)
    labels = torch.tensor(labels)

    # Few-shot reference set
    ref_df = df.groupby('classification', group_keys=False).apply(lambda x: x.sample(n=5, random_state=42)).reset_index(drop=True)
    ref_texts = ref_df['text'].tolist()
    ref_labels = ref_df['classification'].tolist()

    ref_tokens = tokenizer(ref_texts, return_tensors='pt', padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        ref_embeddings = model.encoder(**ref_tokens).last_hidden_state[:, 0, :]

    # Predict via nearest neighbor (cosine similarity)
    preds = []
    for emb in embeddings:
        sims = cosine_similarity(emb.unsqueeze(0), ref_embeddings)
        pred_label = ref_labels[torch.argmax(sims).item()]
        preds.append(pred_label)

    print(classification_report(labels, preds, target_names=['F', 'T']))

    # Reduce embeddings to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(embeddings.numpy())

    # Plot
    plt.figure(figsize=(8, 6))
    colors = ['red' if label == 0 else 'blue' for label in labels]
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.7, label='Embeddings')
    plt.title('Siamese BERT Embedding Space (t-SNE)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(['F (red)', 'T (blue)'])
    plt.grid(True)
    plt.show()

    model.encoder.save_pretrained("app/models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese")
    tokenizer.save_pretrained("app/models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese")


from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

def is_requirement(text, threshold=0.5):
    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained('app/models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese')
    encoder = AutoModel.from_pretrained('app/models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese')
    encoder.eval()

    # Reference texts (few-shot)
    reference_texts = [
        "The system shall provide access to user data.",
        "The application must encrypt stored files.",
        "1. Introduction",
        "Document Revision History"
    ]
    reference_labels = [1, 1, 0, 0]

    # Encode reference texts
    with torch.no_grad():
        ref_inputs = tokenizer(reference_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        ref_embs = encoder(**ref_inputs).last_hidden_state[:, 0, :]

        # Encode input text
        inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
        emb = encoder(**inp).last_hidden_state[:, 0, :]

        # Cosine sim with all references
        sims = cosine_similarity(emb, ref_embs)
        max_sim, max_idx = torch.max(sims, dim=0)

        if max_sim.item() > threshold and reference_labels[max_idx.item()] == 1:
            return True  # is a requirement
        else:
            return False


def test():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("app/models/requirement_identification/bert-base-requirements-identification-generative-ai")

    def predict_if_requirement(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        classification = 'T' if pred == 1 else 'F'

        print(f"Text: {text}\nPrediction: {classification}")

    predict_if_requirement("The system is about a way to provide something to users.")
    predict_if_requirement("The system shall provide a way for the user to input their information.")
    predict_if_requirement("1.2.3")



train_siamese_model()