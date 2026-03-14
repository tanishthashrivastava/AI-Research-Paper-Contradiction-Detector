import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.title("AI Research Paper Analyzer")

data = pd.read_csv("data/papers.csv")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_resource
def load_contradiction_model():
    return pipeline("text-classification", model="facebook/bart-large-mnli")

contradiction_model = load_contradiction_model()

conclusions = data["conclusion"].tolist()
embeddings = embedding_model.encode(conclusions)

st.header("🔎 Semantic Search")

query = st.text_input("Enter research statement")

if st.button("Search Similar Papers"):

    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)

    top_indices = similarities[0].argsort()[-3:][::-1]

    st.subheader("Top Similar Papers")

    for idx in top_indices:
        st.write("Title:", data.iloc[idx]["title"])
        st.write("Conclusion:", data.iloc[idx]["conclusion"])
        st.write("---")

st.header("⚠️ Contradiction Detection")

text1 = st.text_area("Paper 1 Conclusion")
text2 = st.text_area("Paper 2 Conclusion")

if st.button("Check Contradiction"):

    result = contradiction_model(f"{text1} </s> {text2}")

    label = result[0]["label"]
    score = result[0]["score"]

    if label == "CONTRADICTION":
        st.error("These papers contradict each other")
    elif label == "ENTAILMENT":
        st.success("These papers support each other")
    else:
        st.info("Relationship unclear")

    st.write("Confidence Score:", score)