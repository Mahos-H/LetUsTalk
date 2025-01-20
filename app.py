# **Team Let Us Talk Final Notebook**
#### **We used Twitter Setiment Analysis Dataset from kaggle <https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset>**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# SciKit-Learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# WordCloud
from wordcloud import WordCloud

# Deep Learning / Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import gensim.downloader as gensim_api
import pickle  

### **1) LOAD & COMBINE DATASETS**
train_data = pd.read_csv("data/train.csv", encoding='latin1')
test_data  = pd.read_csv("data/test.csv",  encoding='latin1')
df_full = pd.concat([train_data, test_data], ignore_index=True)

### **2) KEEP ONLY text & sentiment**
df = df_full[['text','sentiment']].copy()

### **3) DROP ROWS WITH MISSING text OR sentiment**
df.dropna(subset=['text','sentiment'], inplace=True)
df.drop_duplicates(subset=['text'], inplace=True)
df.reset_index(drop=True, inplace=True)

### **4) MAP SENTIMENT STRINGS TO 0, 1, 2**
sentiment_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
df['sentiment_mapped'] = df['sentiment'].map(sentiment_map)

before_drop = len(df)
df = df.dropna(subset=['sentiment_mapped'])
df.reset_index(drop=True, inplace=True)
after_drop = len(df)

df['sentiment_mapped'] = df['sentiment_mapped'].astype(int)

### **5) ADVANCED TEXT CLEANING**
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def advanced_text_cleaning(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove user mentions, hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation / non-alphanumeric
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    tokens = text.split()
    # Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(advanced_text_cleaning)

### **6) EDA & VISUALIZATIONS**
# (A) Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sentiment')
plt.title("Original Sentiment String Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sentiment_mapped')
plt.title("Mapped Sentiment Distribution (0=neg,1=neu,2=pos)")
plt.show()

print("\nCounts:")
print(df['sentiment_mapped'].value_counts())

# (B) Text Length Distribution
df['text_length'] = df['clean_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(6,4))
sns.histplot(df['text_length'], bins=30, kde=True, color='green')
plt.title("Distribution of Tweet Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# (C) Word Clouds (with checks)
neg_text = " ".join(df.loc[df['sentiment_mapped']==0, 'clean_text'])
neu_text = " ".join(df.loc[df['sentiment_mapped']==1, 'clean_text'])
pos_text = " ".join(df.loc[df['sentiment_mapped']==2, 'clean_text'])

def plot_wordcloud(text_data, title="Word Cloud"):
    if not text_data.strip():
        print(f"No data for {title} (no matching rows). Skipping.")
        return
    wc = WordCloud(width=400, height=400, background_color='white').generate(text_data)
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plot_wordcloud(neg_text, "Negative Word Cloud")
plt.subplot(1,3,2)
plot_wordcloud(neu_text, "Neutral Word Cloud")
plt.subplot(1,3,3)
plot_wordcloud(pos_text, "Positive Word Cloud")
plt.show()

### **7) TRAIN-TEST SPLIT**
X = df['clean_text']
y = df['sentiment_mapped']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
### **8) FEATURE EXTRACTION**
# (A) BAG-OF-WORDS
bow_vectorizer = CountVectorizer(max_features=10000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow  = bow_vectorizer.transform(X_test)

# (B) TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

# (C) PRETRAINED WORD2VEC
print("\nLoading Pretrained Word2Vec Model (Google News)...")
try:
    w2v_model = gensim_api.load("word2vec-google-news-300")  # ~1.6GB
    print("Google News Word2Vec loaded successfully.")
except Exception as e:
    print("Error loading 'word2vec-google-news-300'. Consider using a smaller model.")
    raise e

def text_to_w2v(text):
    tokens = text.split()
    vectors = []
    for t in tokens:
        if t in w2v_model.key_to_index:
            vectors.append(w2v_model[t])
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300, dtype='float32')

def get_w2v_embeddings(text_series):
    return np.vstack(text_series.apply(text_to_w2v).values)

X_train_w2v = get_w2v_embeddings(X_train)
X_test_w2v  = get_w2v_embeddings(X_test)

print("Shapes:")
print("BOW   :", X_train_bow.shape,  "->", X_test_bow.shape)
print("TF-IDF:", X_train_tfidf.shape,"->", X_test_tfidf.shape)
print("W2V   :", X_train_w2v.shape,  "->", X_test_w2v.shape)

### **9) 3D PCA VISUALIZATION ON TF-IDF & WORD2VEC**
from mpl_toolkits.mplot3d import Axes3D

N_SAMPLES = 3000
if X_train_tfidf.shape[0] > N_SAMPLES:
    X_tfidf_samp = X_train_tfidf[:N_SAMPLES]
    y_samp       = y_train[:N_SAMPLES]
    X_w2v_samp   = X_train_w2v[:N_SAMPLES]
else:
    X_tfidf_samp = X_train_tfidf
    y_samp       = y_train
    X_w2v_samp   = X_train_w2v

pca_tfidf = PCA(n_components=3, random_state=42)
pca_w2v   = PCA(n_components=3, random_state=42)

X_tfidf_3d = pca_tfidf.fit_transform(X_tfidf_samp.toarray())
X_w2v_3d   = pca_w2v.fit_transform(X_w2v_samp)

fig = plt.figure(figsize=(12,5))

# TF-IDF
ax = fig.add_subplot(1,2,1, projection='3d')
colors = {0:'red', 1:'blue', 2:'green'}
color_vals = [colors[val] for val in y_samp]
ax.scatter(X_tfidf_3d[:,0], X_tfidf_3d[:,1], X_tfidf_3d[:,2], c=color_vals, alpha=0.6)
ax.set_title("3D PCA (TF-IDF)")

# Word2Vec
ax = fig.add_subplot(1,2,2, projection='3d')
color_vals = [colors[val] for val in y_samp]
ax.scatter(X_w2v_3d[:,0], X_w2v_3d[:,1], X_w2v_3d[:,2], c=color_vals, alpha=0.6)
ax.set_title("3D PCA (Word2Vec)")

plt.show()

def train_and_eval(model, X_tr, X_te, y_tr, y_te, model_name):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"{model_name} Accuracy = {acc:.4f}")
    print(classification_report(y_te, preds))
    ConfusionMatrixDisplay.from_predictions(y_te, preds)
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    return acc

baseline_acc = y_test.value_counts(normalize=True).max()

### **11) DEEP LEARNING MODELS**
VOCAB_SIZE = 20000
MAX_LENGTH = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences  = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_padded  = pad_sequences(test_sequences,  maxlen=MAX_LENGTH, padding='post', truncating='post')

y_train_dl = np.array(y_train)
y_test_dl  = np.array(y_test)

### **(C) BiLSTM + Attention**
class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1),
            initializer="normal", trainable=True
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1),
            initializer="zeros", trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.matmul(inputs, self.W)  
        e = tf.squeeze(e, -1)          
        e = e + tf.squeeze(self.b, -1) 
        alpha = tf.nn.softmax(e)       
        alpha = tf.expand_dims(alpha, axis=-1) 
        context = inputs * alpha       
        context = tf.reduce_sum(context, axis=1) 
        return context

bilstm_att_model = models.Sequential()
bilstm_att_model.add(Embedding(VOCAB_SIZE, 128, input_length=MAX_LENGTH))
bilstm_att_model.add(Bidirectional(LSTM(128, return_sequences=True)))
bilstm_att_model.add(AttentionLayer())
bilstm_att_model.add(Dense(128, activation='relu'))
bilstm_att_model.add(Dropout(0.3))
bilstm_att_model.add(Dense(3, activation='softmax'))

bilstm_att_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n=== BiLSTM + Attention Model Summary ===")
bilstm_att_model.summary()

history_bilstm_att = bilstm_att_model.fit(
    X_train_padded, y_train_dl,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=1
)

bilstm_att_loss, bilstm_att_acc = bilstm_att_model.evaluate(X_test_padded, y_test_dl, verbose=0)
print(f"BiLSTM + Attention Test Accuracy: {bilstm_att_acc:.4f}")
 
### **SAVE THE TRAINED MODEL AND TOKENIZER**
print("\nSaving the BiLSTM+Attention model and tokenizer...")

# 1) Saving the Keras model 
bilstm_att_model.save("models/bilstm_att_model.h5")

# 2) Save the tokenizer using pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")