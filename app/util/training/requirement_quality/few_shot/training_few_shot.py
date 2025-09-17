

import pandas as pd
import torch
from torch import nn
from torch.utils.data import  DataLoader
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from sklearn.metrics import classification_report

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset as TorchDataset
import random

evaluation_class="result_binary_verifiability"
save_path = "app/models/requirement_quality/bert-base-requirements-quality-verifiability-siamese"

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

class SiameseDataset(TorchDataset):
    def __init__(self, df, tokenizer, num_pairs=1000):
        self.pairs = []
        self.labels = []
        self.tokenizer = tokenizer
        self.label_to_indices = df.groupby(evaluation_class).indices
        self.df = df.reset_index(drop=True)

        for _ in range(num_pairs):
            # Positive pair
            label = random.choice(list(self.label_to_indices.keys()))
            idx1, idx2 = random.sample(list(self.label_to_indices[label]), 2)
            self.pairs.append((df.loc[idx1, 'requirement'], df.loc[idx2, 'requirement']))
            self.labels.append(1)

            # Negative pair
            label1, label2 = random.sample(list(self.label_to_indices.keys()), 2)
            idx1 = random.choice(self.label_to_indices[label1])
            idx2 = random.choice(self.label_to_indices[label2])
            self.pairs.append((df.loc[idx1, 'requirement'], df.loc[idx2, 'requirement']))
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
    df = pd.read_excel('app/datasets/requirement_quality/dataset.xlsx')

    df = df[["requirement", evaluation_class]]

    df = df.sample(frac=1).reset_index(drop=True)
    df[evaluation_class] = df[evaluation_class].map({'T': 1, 'F': 0})
    df['requirement'] = df['requirement'].astype(str)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = SiameseBERT('distilbert-base-uncased')
    dataset = SiameseDataset(df, tokenizer, num_pairs=400)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(4):
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
            tokens = tokenizer(batch['requirement'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=32)
            out = model.encoder(**tokens).last_hidden_state[:, 0, :]
            embeddings.append(out)
            labels.extend(batch[evaluation_class].tolist())

    embeddings = torch.cat(embeddings)
    labels = torch.tensor(labels)

    # Few-shot reference set
    ref_df = df.groupby(evaluation_class, group_keys=False).apply(lambda x: x.sample(n=5, random_state=42)).reset_index(drop=True)
    ref_texts = ref_df['requirement'].tolist()
    ref_labels = ref_df[evaluation_class].tolist()

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

    model.encoder.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def predict_all_qualities():
    qualities = ["ambiguity", "feasibility", "singularity", "verifiability"]
    input_df = pd.read_csv("app/datasets/requirement_quality/PURE_train.csv")
    input_df['requirement'] = input_df['requirement'].astype(str)

    for quality in qualities:
        # Load model and tokenizer
        inference_path = f"app/models/requirement_quality/bert-base-requirements-quality-{quality}-siamese"
        tokenizer = AutoTokenizer.from_pretrained(inference_path)
        model = AutoModel.from_pretrained(inference_path)
        model.eval()

        # Load reference examples
        ref_df = pd.read_excel("app/datasets/requirement_quality/dataset.xlsx")
        col = f"result_binary_{quality}"
        ref_df = ref_df[["requirement", col]].dropna()
        ref_df['requirement'] = ref_df['requirement'].astype(str)
        ref_df[col] = ref_df[col].map({'T': 1, 'F': 0})
        samples = ref_df.groupby(col, group_keys=False).apply(lambda x: x.sample(5, random_state=42))
        ref_texts = samples['requirement'].tolist()
        ref_labels = samples[col].tolist()

        # Embed reference set
        with torch.no_grad():
            ref_tokens = tokenizer(ref_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
            ref_embs = model(**ref_tokens).last_hidden_state[:, 0, :]

        # Predict for each requirement
        preds = []
        for req in input_df['requirement']:
            with torch.no_grad():
                tokens = tokenizer(req, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
                emb = model(**tokens).last_hidden_state[:, 0, :]
                sims = cosine_similarity(emb, ref_embs)
                top = torch.topk(sims, k=5).indices
                score = sum(ref_labels[i] for i in top) / 5
                preds.append('T' if round(score) == 1 else 'F')

        input_df[f"result_binary_{quality}"] = preds

    input_df.to_csv("app/datasets/requirement_quality/PURE_with_predictions.csv", index=False)




def test(k=5):

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModel.from_pretrained(save_path)
    model.eval()

    # Load dataset
    df = pd.read_excel('app/datasets/requirement_quality/dataset.xlsx')
    df = df[["requirement", evaluation_class]].dropna()
    df['requirement'] = df['requirement'].astype(str)
    df[evaluation_class] = df[evaluation_class].map({'T': 1, 'F': 0})

    # Sample 5 positive and 5 negative reference examples
    ref_df = (
        df.groupby(evaluation_class, group_keys=False)
        .apply(lambda x: x.sample(n=5, random_state=42))
        .reset_index(drop=True)
    )
    ref_texts = ref_df['requirement'].tolist()
    ref_labels = ref_df[evaluation_class].tolist()

    with torch.no_grad():
        ref_tokens = tokenizer(ref_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        ref_embeddings = model(**ref_tokens).last_hidden_state[:, 0, :]

    def predict_with_reference(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 0, :]
            sims = cosine_similarity(emb, ref_embeddings)
            top_indices = torch.topk(sims, k=k).indices.tolist()
            top_labels = [ref_labels[i] for i in top_indices]
            predicted_label = round(sum(top_labels) / k)
            predicted_class = 'T' if predicted_label == 1 else 'F'

            print(f"\nInput: {text}")
            print(f"Predicted Label: {predicted_class}")
            for i in range(k):
                print(f"  Ref #{i+1} ({sims[top_indices[i]]:.4f}): {ref_texts[top_indices[i]]}")

    # Run test cases
    predict_with_reference("The system is about a way to provide something to users.")
    predict_with_reference("The system shall provide a way for the user to input their information.")
    predict_with_reference("1.2.3")
    predict_with_reference("The system should do everything.")
    predict_with_reference("The system should support everything.")
    predict_with_reference("The system should load the main page. Additionally, the system should do everything.")

# train_siamese_model()
predict_all_qualities()