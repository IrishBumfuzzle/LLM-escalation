#!/usr/bin/env python3
"""
Simulation script for testing the Cost-aware Majority Voting (CaMVo) algorithm
on Amazon product listing titles and embeddings.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

from camvo import CaMVoEstimator, run_annotation_loop

# =====================================================================
# 1. Contextual Accuracy & Mock Query Logic
# =====================================================================

def get_true_accuracy(llm_name: str, embedding: np.ndarray) -> float:
    """
    Simulates context-dependent accuracy for each LLM based on embedding features.
    """
    if llm_name == "gpt-4o":
        # Highly accurate across all products
        return 0.92
    elif llm_name == "claude-3-haiku":
        # Moderately accurate across all products
        return 0.80
    elif llm_name == "gemini-flash":
        # High accuracy if the first embedding dimension is positive
        return 0.88 if embedding[0] > 0 else 0.45
    elif llm_name == "llama-3":
        # High accuracy if the second embedding dimension is positive
        return 0.82 if embedding[1] > 0 else 0.40
    return 0.50


def query_llm_mock(llm_name: str, embedding: np.ndarray, text: str, gold_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates querying an LLM and returning a response with noise based on its true accuracy.
    """
    true_acc = get_true_accuracy(llm_name, embedding)
    
    if random.random() < true_acc:
        return gold_json
    else:
        # Scrambled/noisy response representing error
        return {
            "product_title": f"Noisy prediction by {llm_name}",
            "features": []
        }


# =====================================================================
# 2. Data Loading
# =====================================================================

def load_real_data(csv_path: str, limit: int = 200) -> Tuple[List[str], np.ndarray]:
    """
    Utility to load real Amazon titles and Nomic embeddings from products_with_embeddings.csv.
    """
    print(f"Loading first {limit} rows from '{csv_path}'...")
    df = pd.read_csv(csv_path, nrows=limit)
    product_texts = df['title'].fillna("").tolist()
    
    embeddings_list = []
    for _, row in df.iterrows():
        emb_str = row['embedding']
        emb = json.loads(emb_str)
        embeddings_list.append(emb)
        
    embeddings = np.array(embeddings_list)
    return product_texts, embeddings


# =====================================================================
# 3. Main Runner
# =====================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    # 1. Define LLM pool configuration with costs and priors
    llm_pool = {
        "gpt-4o": {"cost": 0.005, "prior_accuracy": 0.90},
        "claude-3-haiku": {"cost": 0.0005, "prior_accuracy": 0.75},
        "gemini-flash": {"cost": 0.0002, "prior_accuracy": 0.70},
        "llama-3": {"cost": 0.0001, "prior_accuracy": 0.60}
    }

    csv_path = "products_with_embeddings.csv"
    if os.path.exists(csv_path):
        print(f"Found embeddings CSV file '{csv_path}'. Running simulation on real dataset...")
        try:
            product_texts, embeddings = load_real_data(csv_path, limit=200)
            d = embeddings.shape[1]
        except Exception as e:
            print(f"Warning: Failed to load real embeddings: {e}. Falling back to synthetic data.")
            d = 384
            product_texts = [f"Sample Product Title {i}" for i in range(100)]
            embeddings = np.random.randn(100, d)
    else:
        print(f"Embeddings CSV file '{csv_path}' not found. Generating synthetic dataset...")
        d = 384
        product_texts = [f"Sample Product Title {i}" for i in range(100)]
        embeddings = np.random.randn(100, d)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    # 2. Instantiate CaMVo Estimator
    estimator = CaMVoEstimator(
        llm_pool=llm_pool,
        d=d,
        lambda_L=1.0,
        lambda_R=5.0,
        alpha_bandit=0.5
    )

    # 3. Run CaMVo annotation loop with mock query callback
    # Target confidence delta=0.95, enforce min query count k_min=2
    # Query all models during first 10 rounds to initialize parameters
    run_annotation_loop(
        product_texts=product_texts,
        embeddings=embeddings,
        estimator=estimator,
        query_fn=query_llm_mock,
        delta=0.95,
        k_min=2
    )


if __name__ == "__main__":
    main()
