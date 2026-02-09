from numpy.f2py.crackfortran import verbose
from transformers import DataCollatorWithPadding, BertForSequenceClassification, EarlyStoppingCallback, \
    TrainingArguments, Trainer, XLNetForSequenceClassification, XLNetConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from statsmodels.stats.inter_rater import fleiss_kappa

from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import clone_model


QUALITY_FILE_NAME = 'dataset_finalized.xlsx'

def load_and_preprocess(load_quality = True, load_reviewer = True, index = 4, random_state=42):

    # Load dataset
    if(load_quality == False):
        if(load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_identification/reviewer_1/dataset_1_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_2/dataset_2_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_3_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_4_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_4/dataset_5_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_5/dataset_6_unlabeled_req.xlsx"
            ]

            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')

    else:
        if(load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_quality/reviewer_1/dataset_1_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_2/dataset_2_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_3_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_4_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_4/dataset_5_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_5/dataset_6_quality_unlabeled.xlsx"
            ]
            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)

        else:
            df = pd.read_excel('app/datasets/requirement_quality/' + QUALITY_FILE_NAME)

    if(load_quality):
        col0, target_col = df.columns[0], df.columns[index]
    else:
        col0, target_col = df.columns[0], df.columns[1]

    df = df[[col0, target_col]].rename(columns={col0: 'text', target_col: 'classification'})

    # df['classification'] = df['classification'].map({'T': 1, 'F': 0})
    df['classification'] = df['classification'].map({True: 1, False: 0})

    print(df['classification'].value_counts())

    df_majority = df[df['classification'] == df['classification'].value_counts().idxmax()]
    df_minority = df[df['classification'] == df['classification'].value_counts().idxmin()]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=random_state
    )

    df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # # Lemmatization of the various strings
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x.split()]))

    # Preprocessing
    vectorizer = CountVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['classification'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    return X_train, X_test, y_train, y_test, vectorizer




def load_and_preprocess_embedding(load_quality = True, index = 4, random_state=42):

    if(load_quality == False):
        df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    else:
        df = pd.read_excel('app/datasets/requirement_quality/' + QUALITY_FILE_NAME)
        col0, target_col = df.columns[0], df.columns[index]
        df = df[[col0, target_col]].rename(columns={col0: 'text', target_col: 'classification'})

    # Map classification to binary
    df['classification'] = df['classification'].map({'T': 1, 'F': 0})

    # Split into majority and minority classes
    df_majority = df[df['classification'] == df['classification'].value_counts().idxmax()]
    df_minority = df[df['classification'] == df['classification'].value_counts().idxmin()]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=random_state
    )

    df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Lemmatization and tokenization
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(x)]))

    # Tokenize text for Word2Vec
    tokenized_text = df['text'].apply(word_tokenize)

    # Train Word2Vec model (skipgram or CBOW)
    word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, sg=1)

    # Convert sentences to embeddings by averaging the word vectors
    def get_sentence_vector(tokens):
        word_vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        if len(word_vectors) == 0:
            return np.zeros(word2vec_model.vector_size)  # Return a zero vector if no words found
        return np.mean(word_vectors, axis=0)

    df['embeddings'] = df['text'].apply(lambda x: get_sentence_vector(word_tokenize(x)))

    # Prepare the features and labels
    X = np.stack(df['embeddings'].values)
    y = df['classification'].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    return X_train, X_test, y_train, y_test, word2vec_model



def load_and_preprocess_cnn_rnn(vocab_size=1000, max_len=100, load_quality = True, index = 4, load_reviewer=True, random_state=42):

    if(load_quality == False):
        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_identification/reviewer_1/dataset_1_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_2/dataset_2_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_3_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_4_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_4/dataset_5_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_5/dataset_6_unlabeled_req.xlsx"
            ]
            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    else:

        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_quality/reviewer_1/dataset_1_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_2/dataset_2_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_3_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_4_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_4/dataset_5_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_5/dataset_6_quality_unlabeled.xlsx"
            ]

            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_excel('app/datasets/requirement_quality/' + QUALITY_FILE_NAME)

    if(load_quality):
        col0, target_col = df.columns[0], df.columns[index]
    else:
        col0, target_col = df.columns[0], df.columns[1]

    df = df[[col0, target_col]].rename(columns={col0: 'text', target_col: 'classification'})

    # df['classification'] = df['classification'].map({'T': 1, 'F': 0})
    df['classification'] = df['classification'].map({True: 1, False: 0})

    df_majority = df[df['classification'] == df['classification'].value_counts().idxmax()]
    df_minority = df[df['classification'] == df['classification'].value_counts().idxmin()]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=random_state
    )

    df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(df['classification'].value_counts())

    X = df['text'].values
    y = df['classification'].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train_raw)
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=max_len)

    return X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, tokenizer

def visualize(y_test, y_pred, title="Confusion Matrix"):
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    total = np.sum(cm)
    per_class_acc = []

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FP - FN
        acc = (TP + TN) / total
        per_class_acc.append(acc)

    print(per_class_acc)

    labels = ['False', 'True']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def load_raw_data():
    df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    df['classification'] = df['classification'].map({'T': 1, 'F': 0})

    df_majority = df[df['classification'] == df['classification'].value_counts().idxmax()]
    df_minority = df[df['classification'] == df['classification'].value_counts().idxmin()]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    df = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

    df = df.sample(frac=1).reset_index(drop=True)
    texts = df['text'].values
    labels = df['classification'].values
    return texts, labels

def get_linear_regression_model():
    return LinearRegression()

def get_logistic_regression_model():
    return LogisticRegression(max_iter=500)

def get_naive_bayes_model():
    return MultinomialNB()

def get_svm_model():
    return SVC(kernel='linear', probability=True)

def get_random_forest_model():
    return RandomForestClassifier(n_estimators=250)

def get_knn_model():
    return KNeighborsClassifier()

def get_ann_model(input_dim):
    model = Sequential()

    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def get_cnn_model(vocab_size=1000, max_len=100):
    model = Sequential()

    model.add(Embedding(vocab_size, 256, input_length=max_len))

    model.add(Conv1D(32, 4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def get_rnn_model(vocab_size=250, max_len=100):

    model = Sequential()

    model.add(Embedding(vocab_size, 128, input_length=max_len))

    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(8))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def train_kfold(kfold, model, X, y, **params):
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

    results = []
    model_orig = clone_model(model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    for train_index, test_index in kf.split(X):
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]
        model = clone_model(model_orig)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train_k, y_train_k, **params)

        y_pred = model.predict(X_test_k)

        y_pred = (y_pred > 0.5).astype(int)

        report = classification_report(
            y_test_k,
            y_pred,
            output_dict=True,
            target_names=["Class 0", "Class 1"]
        )

        acc = accuracy_score(y_test_k, y_pred)

        results.append({
            "class_0_precision": report["Class 0"]["precision"],
            "class_0_recall": report["Class 0"]["recall"],
            "class_0_f1": report["Class 0"]["f1-score"],
            "accuracy": acc,
            "class_1_precision": report["Class 1"]["precision"],
            "class_1_recall": report["Class 1"]["recall"],
            "class_1_f1": report["Class 1"]["f1-score"],
            "accuracy_1": acc,
        })

        print(classification_report(y_test_k, y_pred, target_names=["Class 0", "Class 1"]))
        df_results = pd.DataFrame(results)

        df_results = pd.concat([df_results], ignore_index=True)

        # Export to Excel
        df_results.to_excel("kfold_results.xlsx", index=False)

        print(report)


def train_linear_regression(kfold=10):

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
    model = get_linear_regression_model()

    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y)
    else:

        # Fit the linear regression model
        model.fit(X_train, y_train)

        # Predict using linear regression. Since it returns probabilities, convert to 0 and 1.
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

    # visualize(y_test, y_pred, "Linear Regression Confusion Matrix")
    #
    # feature_names = vectorizer.get_feature_names_out()
    # coefficients = model.coef_.ravel()  # Ensures it's a flat array
    #
    # top_n = 10
    # top_pos_idx = np.argsort(coefficients)[-top_n:]
    # top_neg_idx = np.argsort(coefficients)[:top_n]
    #
    # print("Top positive influence words (push toward class 1):")
    # for i in reversed(top_pos_idx):
    #     print(f"{feature_names[i]}", end="\t")
    #
    # print("\nTop negative influence words (push toward class 0):")
    # for i in top_neg_idx:
    #     print(f"{feature_names[i]}", end="\t")

        return y_pred


def train_logistic_regression(kfold=10):

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    model = get_logistic_regression_model()

    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y)
    else:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # visualize(y_test, y_pred, "Logistic Regression Confusion Matrix")
        #
        # feature_names = vectorizer.get_feature_names_out()
        # coefficients = model.coef_[0]
        #
        # # Sort by absolute value (most influential)
        # top_n = 10
        # top_positive_indices = np.argsort(coefficients)[-top_n:]
        # top_negative_indices = np.argsort(coefficients)[:top_n]
        #
        # print("Top positive words:")
        # for i in reversed(top_positive_indices):
        #     print(f"{feature_names[i]}", end="\t")
        #
        # print("\nTop negative words:")
        # for i in top_negative_indices:
        #     print(f"{feature_names[i]}", end="\t")

        return y_pred


def train_naive_bayes(kfold=10):

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    model = get_naive_bayes_model()

    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y)
    else:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # visualize(y_test, y_pred, "Naive Bayes Confusion Matrix")
        #
        # feature_names = vectorizer.get_feature_names_out()
        # log_probs = model.feature_log_prob_
        #
        # # Difference in log-probabilities between class 1 and class 0
        # log_prob_diff = log_probs[1] - log_probs[0]
        #
        # top_n = 10
        # top_pos_idx = np.argsort(log_prob_diff)[-top_n:]
        # top_neg_idx = np.argsort(log_prob_diff)[:top_n]
        #
        # print("Top words pushing toward class 1:")
        # for i in reversed(top_pos_idx):
        #     print(f"{feature_names[i]}", end="\t")
        #
        # print("\nTop words pushing toward class 0:")
        # for i in top_neg_idx:
        #     print(f"{feature_names[i]}", end="\t")

        return y_pred

def train_svm(kfold=10):

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    model = get_svm_model()
    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y)
    else:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # visualize(y_test, y_pred, "SVM Confusion Matrix")
        #
        # coefficients = model.coef_.ravel()
        # feature_names = vectorizer.get_feature_names_out()
        #
        # top_n = 10
        # top_pos_idx = np.argsort(coefficients)[-top_n:]
        # top_neg_idx = np.argsort(coefficients)[:top_n]
        #
        # print("Top words pushing toward class 1:")
        # for i in reversed(top_pos_idx):
        #     print(f"{feature_names[i]}", end="\t")
        #
        # print("\nTop words pushing toward class 0:")
        # for i in top_neg_idx:
        #     print(f"{feature_names[i]}", end="\t")

        return y_pred

def train_random_forest(kfold=10):
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    model = get_random_forest_model()
    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y)
    else:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # visualize(y_test, y_pred, "Random Forest Confusion Matrix")

        return y_pred

def train_knn(kfold=10):
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    model = get_knn_model()
    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y)
    else:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # visualize(y_test, y_pred, "KNN Confusion Matrix")

        return y_pred


def train_ann(kfold=10):

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    input_dim = X_train.shape[1]

    model = get_ann_model(input_dim)

    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y, epochs=25, batch_size=8, verbose=2)
    else:

        model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=2)

        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # visualize(y_test, y_pred, "ANN Confusion Matrix")

        return y_pred

def train_cnn(kfold=10, vocab_size=1000, max_len=500):

    X_train, X_test, y_train, y_test, _, _, vectorizer = load_and_preprocess_cnn_rnn()

    model = get_cnn_model()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y, epochs=16, batch_size=8, verbose=1, validation_data=(X_test, y_test))
    else:

        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1, validation_data=(X_test, y_test))

        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # visualize(y_test, y_pred, "CNN Confusion Matrix")

        return y_pred

def train_rnn(kfold=10, vocab_size=1000, max_len=100):

    X_train, X_test, y_train, y_test, _, _, tokenizer = load_and_preprocess_cnn_rnn(vocab_size, max_len)

    model = get_rnn_model()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if(kfold > 0):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_kfold(kfold, model, X, y, epochs=20, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    else:

        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_test, y_test))

        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # visualize(y_test, y_pred, "RNN Confusion Matrix")

        return y_pred

def train_meta_random_forest_prediction():
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    model = get_random_forest_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    # visualize(y_test, y_pred, "Random Forest Confusion Matrix")

    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_pred)

    # Get word-level coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = meta_model.coef_[0]

    top_n = 10
    top_positive_indices = np.argsort(coefficients)[-top_n:]
    top_negative_indices = np.argsort(coefficients)[:top_n]

    print("Top positive words:")
    for i in reversed(top_positive_indices):
        print(f"{feature_names[i]}", end="\t")

    print("\nTop negative words:")
    for i in top_negative_indices:
        print(f"{feature_names[i]}", end="\t")

def train_meta_knn_prediction():
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    model = get_knn_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    # visualize(y_test, y_pred, "KNN Confusion Matrix")

    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_pred)

    # Get word-level coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = meta_model.coef_[0]

    top_n = 10
    top_positive_indices = np.argsort(coefficients)[-top_n:]
    top_negative_indices = np.argsort(coefficients)[:top_n]

    print("Top positive words:")
    for i in reversed(top_positive_indices):
        print(f"{feature_names[i]}", end="\t")

    print("\nTop negative words:")
    for i in top_negative_indices:
        print(f"{feature_names[i]}", end="\t")

def train_meta_ann_prediction():

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    input_dim = X_train.shape[1]

    model = get_ann_model(input_dim)

    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

    y_pred = model.predict(X_train)
    y_pred = (y_pred > 0.5).astype(int)

    # visualize(y_test, y_pred, "ANN Confusion Matrix")

    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_pred)

    # Get word-level coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = meta_model.coef_[0]

    top_n = 10
    top_positive_indices = np.argsort(coefficients)[-top_n:]
    top_negative_indices = np.argsort(coefficients)[:top_n]

    print("Top positive words:")
    for i in reversed(top_positive_indices):
        print(f"{feature_names[i]}", end="\t")

    print("\nTop negative words:")
    for i in top_negative_indices:
        print(f"{feature_names[i]}", end="\t")

def train_meta_cnn_prediction():
    # RNN training
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, tokenizer = load_and_preprocess_cnn_rnn(1000, 100)

    model = get_cnn_model()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    model_preds = model.predict(X_train)
    model_labels = (model_preds > 0.5).astype(int).ravel()

    # TF-IDF feature extraction for logistic regression

    vectorizer = CountVectorizer(max_features=1000)
    X_cv = vectorizer.fit_transform(X_train_raw)

    meta_model = LogisticRegression()
    meta_model.fit(X_cv, model_labels)

    # Get word-level coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = meta_model.coef_[0]

    top_n = 10
    top_positive_indices = np.argsort(coefficients)[-top_n:]
    top_negative_indices = np.argsort(coefficients)[:top_n]

    print("Top positive words:")
    for i in reversed(top_positive_indices):
        print(f"{feature_names[i]}", end="\t")

    print("\nTop negative words:")
    for i in top_negative_indices:
        print(f"{feature_names[i]}", end="\t")

def train_meta_rnn_prediction():
    # RNN training
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, tokenizer = load_and_preprocess_cnn_rnn(1000, 100)

    model = get_rnn_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=4, batch_size=64, verbose=1)

    model_preds = model.predict(X_train)
    model_labels = (model_preds > 0.5).astype(int).ravel()

    # TF-IDF feature extraction for logistic regression

    vectorizer = CountVectorizer(max_features=1000)
    X_cv = vectorizer.fit_transform(X_train_raw)

    meta_model = LogisticRegression()
    meta_model.fit(X_cv, model_labels)


    # Get word-level coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = meta_model.coef_[0]

    top_n = 10
    top_positive_indices = np.argsort(coefficients)[-top_n:]
    top_negative_indices = np.argsort(coefficients)[:top_n]

    print("Top positive words:")
    for i in reversed(top_positive_indices):
        print(f"{feature_names[i]}", end="\t")

    print("\nTop negative words:")
    for i in top_negative_indices:
        print(f"{feature_names[i]}", end="\t")

def train_meta_bert_prediction():
    # Load tokenizer and fine-tuned BERT model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "app/models/requirement_identification/bert-base-requirements-identification-generative-ai"
    )
    model.eval()

    # Load dataset
    df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    df['classification'] = df['classification'].map({'T': 1, 'F': 0})
    texts = df['text'].values

    # Get BERT predictions
    preds = []
    with torch.no_grad():
        for text in texts:
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
            preds.append(pred)

    preds = np.array(preds)

    # Vectorize the raw text
    vectorizer = CountVectorizer(max_features=1000)
    X_cv = vectorizer.fit_transform(texts)

    # Train logistic regression to mimic BERT predictions
    meta_model = LogisticRegression()
    meta_model.fit(X_cv, preds)

    # Interpret coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = meta_model.coef_[0]

    top_n = 10
    top_positive_indices = np.argsort(coefficients)[-top_n:]
    top_negative_indices = np.argsort(coefficients)[:top_n]

    print("Top positive words (most predictive of requirement):")
    for i in reversed(top_positive_indices):
        print(f"{feature_names[i]}", end="\t")

    print("\nTop negative words (least predictive or predictive of non-requirement):")
    for i in top_negative_indices:
        print(f"{feature_names[i]}", end="\t")

def train_ensemble(vocab_size=5000, max_len=100, random_state=42):
    # Preprocess data for traditional models and deep learning models
    X_train_traditional, X_test_traditional, y_train, y_test, vectorizer = load_and_preprocess(random_state=random_state)
    X_train_cnn_rnn, X_test_cnn_rnn, _, _, _, _, tokenizer = load_and_preprocess_cnn_rnn(vocab_size, max_len, random_state=random_state)

    # Define models
    models_traditional = {
        "linear_regression": get_linear_regression_model(),
        "logistic_regression": get_logistic_regression_model(),
        "naive_bayes": get_naive_bayes_model(),
        "support_vector_machine": get_svm_model(),
        "random_forest": get_random_forest_model()
    }
    models_dl = {
        "cnn": get_cnn_model(),
        "rnn": get_rnn_model()
    }

    # Train traditional models and collect predictions
    train_predictions = {}
    test_predictions = {}

    for name, model in models_traditional.items():
        model.fit(X_train_traditional, y_train)
        if hasattr(model, "predict_proba"):
            train_predictions[name] = model.predict_proba(X_train_traditional)[:, 1]
            test_predictions[name] = model.predict_proba(X_test_traditional)[:, 1]
        else:
            train_predictions[name] = np.clip(model.predict(X_train_traditional), 0, 1)
            test_predictions[name] = np.clip(model.predict(X_test_traditional), 0, 1)

        print(name)
        print(classification_report(y_test, (test_predictions[name] > 0.5).astype(int)))


    # Train deep learning models and collect predictions
    for name, model in models_dl.items():
        if name == 'cnn':
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_cnn_rnn, y_train, epochs=25, batch_size=32, verbose=1)
        else:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_cnn_rnn, y_train, epochs=25, batch_size=32, verbose=1)

        train_predictions[name] = model.predict(X_train_cnn_rnn).flatten()
        test_predictions[name] = model.predict(X_test_cnn_rnn).flatten()

    # Stack predictions for training and testing
    stacked_train = np.stack([train_predictions[name] for name in train_predictions.keys()], axis=1)
    stacked_test = np.stack([test_predictions[name] for name in test_predictions.keys()], axis=1)

    # --- Average Ensemble ---
    avg_probs_test = np.mean(stacked_test, axis=1)
    avg_ensemble_preds = (avg_probs_test > 0.5).astype(int)
    print("Average Ensemble")

    # --- Majority Vote Ensemble ---
    binary_preds_test = (stacked_test > 0.5).astype(int)
    majority_vote_preds = (np.sum(binary_preds_test, axis=1) > (binary_preds_test.shape[1] / 2)).astype(int)
    print("Majority Vote Ensemble")

    # Train and evaluate Logistic Regression meta-model
    meta_model_lr = LogisticRegression()
    meta_model_lr.fit(stacked_train, y_train)
    ensemble_preds_lr = meta_model_lr.predict(stacked_test)
    print("Logistic Regression Meta-Model")

    # Train and evaluate Gradient Boosting meta-model
    meta_model_gb = GradientBoostingClassifier()
    meta_model_gb.fit(stacked_train, y_train)
    ensemble_preds_gb = meta_model_gb.predict(stacked_test)
    print("Gradient Boosting Meta-Model")
    print(classification_report(y_test, ensemble_preds_gb))

    return {
        "average_ensemble": classification_report(y_test, avg_ensemble_preds, target_names=["Class 0", "Class 1"], output_dict=True),
        "majority_vote_ensemble": classification_report(y_test, majority_vote_preds, target_names=["Class 0", "Class 1"], output_dict=True),
        "logistic_regression_meta_model": classification_report(y_test, ensemble_preds_lr, target_names=["Class 0", "Class 1"], output_dict=True),
        "gradient_boost_meta_model": classification_report(y_test, ensemble_preds_gb, target_names=["Class 0", "Class 1"], output_dict=True)
    }




def train_bert(load_quality = True, index = 4, load_reviewer = True, random_state=42):


    if(load_quality == False):
        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_identification/reviewer_1/dataset_1_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_2/dataset_2_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_3_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_4_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_4/dataset_5_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_5/dataset_6_unlabeled_req.xlsx"
            ]
            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    else:

        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_quality/reviewer_1/dataset_1_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_2/dataset_2_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_3_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_4_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_4/dataset_5_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_5/dataset_6_quality_unlabeled.xlsx"
            ]

            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_excel('app/datasets/requirement_quality/' + QUALITY_FILE_NAME)

    if(load_quality):
        col0, target_col = df.columns[0], df.columns[index]
    else:
        col0, target_col = df.columns[0], df.columns[1]
    df = df[[col0, target_col]].rename(columns={col0: 'text', target_col: 'classification'})

    print(df['classification'].value_counts())

    df['classification'] = df['classification'].map({True: 1, False: 0})
    # df['classification'] = df['classification'].map({'T': 1, 'F': 0})

    # Rebalance

    minority_class = df[df['classification'] == df['classification'].value_counts().idxmin()]
    majority_class = df[df['classification'] == df['classification'].value_counts().idxmax()]

    # Oversample the minority class
    oversampled_minority = minority_class.sample(len(majority_class), replace=True)

    # Combine to get the balanced dataset
    df = pd.concat([majority_class, oversampled_minority], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    df['text'] = df['text'].astype(str)

    print(df['classification'].value_counts())

    # Randomize the entries
    # df = df.sample(len(df) * 5, random_state=35, replace=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
    #                                                       hidden_dropout_prob=0.2,
    #                                                       attention_probs_dropout_prob=0.2,
    #                                                       num_labels=2)

    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                          hidden_dropout_prob=0.2,
                                                          attention_probs_dropout_prob=0.2,
                                                          num_labels=2)

    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=32,
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.05,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="eval_accuracy",
        logging_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_predictions = trainer.predict(eval_dataset)
    test_labels = test_predictions.label_ids
    test_preds = np.argmax(test_predictions.predictions, axis=1)

    print("Classification Report for Test Data")
    # visualize(test_labels, test_preds, "BERT Confusion Matrix")
    print(classification_report(test_labels, test_preds, target_names=["Class 0 (F)", "Class 1 (T)"]))

    model.save_pretrained("../models/bert-base-requirements-identification-generative-ai")
    return classification_report(test_labels, test_preds, target_names=["Class 0", "Class 1"], output_dict=True)


def train_xl_net(load_quality = True, index = 4, load_reviewer = True, random_state=42):


    if(load_quality == False):
        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_identification/reviewer_1/dataset_1_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_2/dataset_2_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_3_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_4_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_4/dataset_5_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_5/dataset_6_unlabeled_req.xlsx"
            ]
            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    else:

        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_quality/reviewer_1/dataset_1_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_2/dataset_2_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_3_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_4_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_4/dataset_5_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_5/dataset_6_quality_unlabeled.xlsx"
            ]

            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_excel('app/datasets/requirement_quality/' + QUALITY_FILE_NAME)

    if(load_quality):
        col0, target_col = df.columns[0], df.columns[index]
    else:
        col0, target_col = df.columns[0], df.columns[1]
    df = df[[col0, target_col]].rename(columns={col0: 'text', target_col: 'classification'})

    print(df['classification'].value_counts())

    df['classification'] = df['classification'].map({True: 1, False: 0})
    # df['classification'] = df['classification'].map({'T': 1, 'F': 0})

    # Rebalance

    minority_class = df[df['classification'] == df['classification'].value_counts().idxmin()]
    majority_class = df[df['classification'] == df['classification'].value_counts().idxmax()]

    # Oversample the minority class
    oversampled_minority = minority_class.sample(len(majority_class), replace=True)

    # Combine to get the balanced dataset
    df = pd.concat([majority_class, oversampled_minority], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    df['text'] = df['text'].astype(str)

    print(df['classification'].value_counts())

    # Randomize the entries
    df = df.sample(len(df) * 1, random_state=random_state, replace=True)

    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
    #                                                       hidden_dropout_prob=0.2,
    #                                                       attention_probs_dropout_prob=0.2,
    #                                                       num_labels=2)

    config = XLNetConfig.from_pretrained("xlnet-base-cased", hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2, num_labels=2)
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",
                                                          config=config)

    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=32,
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.05,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="eval_accuracy",
        logging_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_predictions = trainer.predict(eval_dataset)
    test_labels = test_predictions.label_ids
    test_preds = np.argmax(test_predictions.predictions, axis=1)

    print("Classification Report for Test Data")
    # visualize(test_labels, test_preds, "BERT Confusion Matrix")
    print(classification_report(test_labels, test_preds, target_names=["Class 0 (F)", "Class 1 (T)"]))

    model.save_pretrained("../models/bert-base-requirements-identification-generative-ai")
    return classification_report(test_labels, test_preds, target_names=["Class 0", "Class 1"], output_dict=True)

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

def train_bert_few_shot(load_quality = True, index = 4, load_reviewer = True, random_state=42):

    if(load_quality == False):
        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_identification/reviewer_1/dataset_1_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_2/dataset_2_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_3_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_4_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_4/dataset_5_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_5/dataset_6_unlabeled_req.xlsx"
            ]
            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    else:

        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_quality/reviewer_1/dataset_1_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_2/dataset_2_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_3_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_4_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_4/dataset_5_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_5/dataset_6_quality_unlabeled.xlsx"
            ]

            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_excel('app/datasets/requirement_quality/' + QUALITY_FILE_NAME)

    if(load_quality):
        col0, target_col = df.columns[0], df.columns[index]
    else:
        col0, target_col = df.columns[0], df.columns[1]
    df = df[[col0, target_col]].rename(columns={col0: 'text', target_col: 'classification'})

    print(df['classification'].value_counts())

    df['classification'] = df['classification'].map({True: 1, False: 0})
    # df['classification'] = df['classification'].map({'T': 1, 'F': 0})

    df['text'] = df['text'].astype(str)

    # Few-shot sampling
    few_shot_df = df.groupby('classification', group_keys=False).apply(
        lambda x: x.sample(n=40, random_state=random_state)
    ).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoder = AutoModel.from_pretrained("distilbert-base-uncased")

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
    for epoch in range(6):
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
    return classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], output_dict=True)

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

def train_siamese_model(load_quality = True, index = 4, load_reviewer = True, random_state=42):

    if(load_quality == False):
        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_identification/reviewer_1/dataset_1_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_2/dataset_2_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_3_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_3/dataset_4_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_4/dataset_5_unlabeled_req.xlsx",
                "app/util/validation/datasets/requirement_identification/reviewer_5/dataset_6_unlabeled_req.xlsx"
            ]
            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_csv('app/datasets/requirement_identification/PURE_train.csv')
    else:

        if (load_reviewer):
            files = [
                "app/util/validation/datasets/requirement_quality/reviewer_1/dataset_1_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_2/dataset_2_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_3_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_3/dataset_4_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_4/dataset_5_quality_unlabeled.xlsx",
                "app/util/validation/datasets/requirement_quality/reviewer_5/dataset_6_quality_unlabeled.xlsx"
            ]

            df = pd.concat((pd.read_excel(f) for f in files), ignore_index=True)
        else:
            df = pd.read_excel('app/datasets/requirement_quality/' + QUALITY_FILE_NAME)

    if(load_quality):
        col0, target_col = df.columns[0], df.columns[index]
    else:
        col0, target_col = df.columns[0], df.columns[1]
    df = df[[col0, target_col]].rename(columns={col0: 'text', target_col: 'classification'})

    print(df['classification'].value_counts())

    df = df.sample(frac=1).reset_index(drop=True)
    df['classification'] = df['classification'].map({True: 1, False: 0})
    # df['classification'] = df['classification'].map({'T': 1, 'F': 0})

    df['text'] = df['text'].astype(str)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = SiameseBERT('distilbert-base-uncased')
    dataset = SiameseDataset(df, tokenizer, num_pairs=100)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(15):
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
    ref_df = df.groupby('classification', group_keys=False).apply(lambda x: x.sample(n=5, random_state=random_state)).reset_index(drop=True)
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

    # visualize(labels, preds, "Siamese BERT Confusion Matrix")
    # Reduce embeddings to 2D using t-SNE
    # tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    # emb_2d = tsne.fit_transform(embeddings.numpy())

    # Plot
    # plt.figure(figsize=(8, 6))
    # colors = ['blue' if label == 0 else 'red' for label in labels]
    # plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.7, label='Embeddings')
    # plt.title('Siamese BERT Embedding Space (t-SNE)')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.grid(True)
    # plt.show()

    # model.encoder.save_pretrained("app/models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese")
    # tokenizer.save_pretrained("app/models/requirement_identification/bert-base-requirements-identification-generative-ai-siamese")
    return classification_report(labels, preds, target_names=["Class 0", "Class 1"], output_dict=True)


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


def derive_fleiss_kappa():
    y_pred_logistic = train_logistic_regression().ravel()
    y_pred_svm = train_svm().ravel()
    y_pred_random_forest = train_random_forest().ravel()
    y_pred_naive_bayes = train_naive_bayes().ravel()
    y_pred_knn = train_knn().ravel()
    y_pred_ann = train_ann().ravel()
    y_pred_cnn = train_cnn().ravel()
    y_pred_rnn = train_rnn().ravel()

    # Collect predictions into (n_samples, n_models)
    preds = np.vstack([y_pred_logistic, y_pred_svm, y_pred_random_forest, y_pred_naive_bayes, y_pred_knn, y_pred_ann, y_pred_cnn, y_pred_rnn]).T

    # Build ratings matrix (rows = samples, cols = [count of 0s, count of 1s])
    ratings_matrix = np.zeros((preds.shape[0], 2), dtype=int)
    for i, row in enumerate(preds):
        counts = np.bincount(row, minlength=2)  # count 0s and 1s
        ratings_matrix[i] = counts

    return fleiss_kappa(ratings_matrix)

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


# print(derive_fleiss_kappa())

# train_rnn()

# Store results for each ensemble type
# results = {
#     "average_ensemble": [],
#     "majority_vote_ensemble": [],
#     "logistic_regression_meta_model": [],
#     "gradient_boost_meta_model": []
# }
#
# # Run 10 iterations
# for i in range(10):
#     reports = train_ensemble(random_state=i + 42)  # vary seed slightly for randomness
#
#     # Loop through each model report
#     for model_name, report in reports.items():
#         # Collect metrics for binary classes
#         results[model_name].append({
#             "class_0_precision": report["Class 0"]["precision"],
#             "class_0_recall": report["Class 0"]["recall"],
#             "class_0_f1": report["Class 0"]["f1-score"],
#             "accuracy": report["accuracy"],
#             "class_1_precision": report["Class 1"]["precision"],
#             "class_1_recall": report["Class 1"]["recall"],
#             "class_1_f1": report["Class 1"]["f1-score"],
#             "accuracy_1": report["accuracy"]
#         })
#
# # Export each to Excel
# for model_name, model_results in results.items():
#     df = pd.DataFrame(model_results)
#     filename = f"{model_name}_results.xlsx"
#     df.to_excel(filename, index=False)
#     print(f"Saved: {filename}")

results = []
for i in range(10):
    report = train_bert(random_state=42 + i)
    results.append({
        "class_0_precision": report["Class 0"]["precision"],
        "class_0_recall": report["Class 0"]["recall"],
        "class_0_f1": report["Class 0"]["f1-score"],
        "accuracy": report["accuracy"],
        "class_1_precision": report["Class 1"]["precision"],
        "class_1_recall": report["Class 1"]["recall"],
        "class_1_f1": report["Class 1"]["f1-score"],
        "accuracy_1": report["accuracy"]
    })

df = pd.DataFrame(results)
filename = f"bert_results.xlsx"
df.to_excel(filename, index=False)
print(f"Saved: {filename}")


results = []
for i in range(10):
    report = train_xl_net(random_state=42 + i)
    results.append({
        "class_0_precision": report["Class 0"]["precision"],
        "class_0_recall": report["Class 0"]["recall"],
        "class_0_f1": report["Class 0"]["f1-score"],
        "accuracy": report["accuracy"],
        "class_1_precision": report["Class 1"]["precision"],
        "class_1_recall": report["Class 1"]["recall"],
        "class_1_f1": report["Class 1"]["f1-score"],
        "accuracy_1": report["accuracy"]
    })

df = pd.DataFrame(results)
filename = f"xlnet_results.xlsx"
df.to_excel(filename, index=False)
print(f"Saved: {filename}")


results = []
for i in range(10):
    report = train_bert_few_shot(random_state=42 + i)
    results.append({
        "class_0_precision": report["Class 0"]["precision"],
        "class_0_recall": report["Class 0"]["recall"],
        "class_0_f1": report["Class 0"]["f1-score"],
        "accuracy": report["accuracy"],
        "class_1_precision": report["Class 1"]["precision"],
        "class_1_recall": report["Class 1"]["recall"],
        "class_1_f1": report["Class 1"]["f1-score"],
        "accuracy_1": report["accuracy"]
    })

df = pd.DataFrame(results)
filename = f"few_shot_results.xlsx"
df.to_excel(filename, index=False)
print(f"Saved: {filename}")


results = []
for i in range(10):
    report = train_siamese_model(random_state=42 + i)
    results.append({
    "class_0_precision": report["Class 0"]["precision"],
        "class_0_recall": report["Class 0"]["recall"],
        "class_0_f1": report["Class 0"]["f1-score"],
        "accuracy": report["accuracy"],
        "class_1_precision": report["Class 1"]["precision"],
        "class_1_recall": report["Class 1"]["recall"],
        "class_1_f1": report["Class 1"]["f1-score"],
        "accuracy_1": report["accuracy"]
    })

df = pd.DataFrame(results)
filename = f"siamese_results.xlsx"
df.to_excel(filename, index=False)
print(f"Saved: {filename}")