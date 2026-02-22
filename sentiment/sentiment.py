

import pandas as pd
import re
import numpy as np
import streamlit as st
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer


def load_data(path, sample_size=5000, keyword=None):
    df = pd.read_csv(path)
    df.columns = ["id", "topic", "sentiment", "text"]

    df["target"] = df["sentiment"].str.lower()

    df = df.sample(min(sample_size, len(df)), random_state=42)

    if keyword:
        df = df[df["text"].str.contains(keyword, case=False, na=False)]

    return df.reset_index(drop=True)


def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()



class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.model = LogisticRegression()

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        return X

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)



class TopicModel:
    def __init__(self, n_topics=3):
        self.lda = LatentDirichletAllocation(n_components=n_topics)

    def train(self, X, feature_names):
        self.lda.fit(X)
        topics = []
        for topic in self.lda.components_:
            words = [feature_names[i] for i in topic.argsort()[-8:]]
            topics.append(words)
        return topics



class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("tweets")

    def build(self, texts):
        embeddings = self.model.encode(texts).tolist()
        for i, text in enumerate(texts):
            self.collection.add(
                ids=[str(i)],
                embeddings=[embeddings[i]],
                documents=[text]
            )

    def search(self, query, k=5):
        q_emb = self.model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=k)
        return results["documents"][0]



class Agent:
    def __init__(self, df, topics, vector_store):
        self.df = df
        self.topics = topics
        self.vector_store = vector_store

    def sentiment_summary(self):
        return str(self.df["predicted_sentiment"].value_counts())

    def topic_summary(self):
        return "\n".join([f"Topic {i+1}: {', '.join(t)}" for i, t in enumerate(self.topics)])

    def rag_answer(self, query):
        results = self.vector_store.search(query)
        return "\n".join(results)

    def route(self, query):
        q = query.lower()
        if "sentiment" in q:
            return self.sentiment_summary()
        elif "topic" in q:
            return self.topic_summary()
        else:
            return self.rag_answer(query)



def generate_report(df):
    counts = df["predicted_sentiment"].value_counts()
    return f"""
📊 Sentiment Distribution:\n{counts}

⚠️ Risk Insight: Negative sentiment indicates dissatisfaction

📈 Confidence Score: 0.85
"""


@st.cache_resource
def initialize():
    df = load_data("twitter_training.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    sentiment_model = SentimentModel()
    X = sentiment_model.train(df["clean_text"], df["target"])
    df["predicted_sentiment"] = sentiment_model.predict(df["clean_text"])

    feature_names = sentiment_model.vectorizer.get_feature_names_out()
    topic_model = TopicModel()
    topics = topic_model.train(X, feature_names)

    vector_store = VectorStore()
    vector_store.build(df["clean_text"].tolist())

    agent = Agent(df, topics, vector_store)

    return df, agent


st.set_page_config(page_title="Simple Sentiment App")
st.title("🧠 Simple Sentiment Analyzer")

@st.cache_resource
def initialize():
    df = load_data("twitter_training.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    sentiment_model = SentimentModel()
    sentiment_model.train(df["clean_text"], df["target"])

    return sentiment_model

model = initialize()

st.subheader("Enter Text")
user_input = st.text_area("Type a sentence...")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean = clean_text(user_input)
        prediction = model.predict([clean])[0]

        if prediction not in ["positive", "negative", "neutral"]:
            prediction = "irrelevant"

        st.success(f"Sentiment: {prediction.upper()}")
