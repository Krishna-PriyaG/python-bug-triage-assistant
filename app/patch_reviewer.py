# app/patch_reviewer.py

import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    classifier = joblib.load(
        "data/processed/minilm_svm_classifier.joblib"
    )

    return embedding_model, classifier

@st.cache_data
def load_data():
    bugs_df = pd.read_csv(
        "data/processed/bug_dataset_ml.csv"
    )

    historical_embeddings = np.load(
        "data/processed/minilm_patch_embeddings.npy"
    )

    return bugs_df, historical_embeddings


bugs_df, historical_embeddings = load_data()


embedding_model, classifier = load_models()
tokenizer = embedding_model.tokenizer

def embed_patch(
    text,
    model,
    tokenizer,
    chunk_size=240,
    overlap=40
):
    text = str(text)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False
    )

    chunk_embeddings = []
    step = chunk_size - overlap

    for start in range(0, len(token_ids), step):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        embedding = model.encode(
            chunk_text,
            convert_to_numpy=True
        )

        chunk_embeddings.append(embedding)

        if end >= len(token_ids):
            break

    return np.mean(chunk_embeddings, axis=0)

def retrieve_similar_bugs(
    query_patch,
    bugs_df,
    historical_embeddings,
    model,
    tokenizer,
    top_k=3
):
    query_embedding = embed_patch(
        query_patch,
        model,
        tokenizer
    )

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        historical_embeddings
    )[0]

    top_indices = similarities.argsort()[::-1][:top_k]

    results = bugs_df.iloc[top_indices].copy()

    results["similarity_score"] = similarities[top_indices]

    return results

def build_rag_context(retrieved_bugs):
    context_parts = []

    for rank, (_, bug) in enumerate(
        retrieved_bugs.iterrows(),
        start=1
    ):
        context_parts.append(
            f"""
            HISTORICAL EXAMPLE {rank}

            Project: {bug['project']}
            Category: {bug['bug_category']}
            Reason: {bug['label_reason']}
            Modified Files: {bug['modified_files']}
            Similarity: {bug['similarity_score']:.3f}
            """
        )

    return "\n".join(context_parts)


def build_review_prompt(
    patch,
    predicted_category,
    rag_context
):
    return f"""
            You are a Python code review assistant.

            Review the user's proposed code patch.

            The ML classifier prediction and retrieved historical examples are
            supporting signals only. They may be incorrect.

            USER PATCH:
            {patch}

            ML CLASSIFIER SIGNAL:
            Predicted bug category: {predicted_category}

            SIMILAR HISTORICAL BUG FIXES:
            {rag_context}

            Provide:

            1. What the patch changes
            2. Likely bug/root cause being addressed
            3. Most likely bug category
            4. Whether the change appears correct
            5. Potential edge cases or risks
            6. Recommended improvements or tests

            Base your conclusions primarily on the user's patch.
            Do not assume retrieved examples have the same root cause.
            """.strip()

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

st.title("Repository-Aware Python Patch Reviewer")

patch = st.text_area(
    "Paste your Python patch",
    height=300
)

if st.button("Analyze Patch"):
    if not patch.strip():
        st.error("Please enter a patch.")

    else:
        with st.spinner("Analyzing patch..."):

            patch_embedding = embed_patch(
                patch,
                embedding_model,
                tokenizer
            )

            predicted_category = classifier.predict(
                patch_embedding.reshape(1, -1)
            )[0]

            similar_bugs = retrieve_similar_bugs(
                patch,
                bugs_df,
                historical_embeddings,
                embedding_model,
                tokenizer,
                top_k=3
            )

            rag_context = build_rag_context(similar_bugs)

            review_prompt = build_review_prompt(
                patch,
                predicted_category,
                rag_context
            )

            response = client.responses.create(
                model="gpt-5.6-luna",
                input=review_prompt
            )

            review = response.output_text

        st.subheader("Predicted Bug Category")
        st.write(predicted_category)

        st.subheader("Similar Historical Bugs")

        st.dataframe(
             similar_bugs[
                    [
                        "project",
                        "bug_id",
                        "bug_category",
                        "similarity_score"
                    ]
            ],
            hide_index=True
        )

        st.subheader("Patch")
        st.code(patch, language="diff")

        st.subheader("AI Patch Review")
        st.markdown(review)