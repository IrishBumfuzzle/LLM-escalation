#!/usr/bin/env python3
"""
Cost-aware Majority Voting (CaMVo) algorithm for LLM-based online dataset annotation.
Implements dynamic, contextual LLM selection and weighting for NER using LinUCB,
Bayesian Beta shape estimation, and Laplace-smoothed confidence regularizers.
"""

import re
import json
from typing import List, Dict, Tuple, Any, Callable

import numpy as np
from scipy.stats import beta as scipy_beta


# =====================================================================
# 1. Helper Functions for NER Metrics & Aggregation
# =====================================================================

def compute_token_f1(pred: dict | str | Any, gold: dict | str | Any) -> float:
    """
    Computes the token-level F1 score between two predictions.
    Supports either dictionary objects (NER output) or raw text.
    """
    if isinstance(pred, dict):
        pred_str = json.dumps(pred, sort_keys=True)
    else:
        pred_str = str(pred)

    if isinstance(gold, dict):
        gold_str = json.dumps(gold, sort_keys=True)
    else:
        gold_str = str(gold)

    # Tokenize by alphanumeric sequences (words/tokens)
    pred_tokens = re.findall(r'\w+', pred_str.lower())
    gold_tokens = re.findall(r'\w+', gold_str.lower())

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    # Count frequencies
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common_tokens = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_counts[t], gold_counts[t]) for t in common_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1


def fuzzy_majority_vote(responses: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
    """
    Performs a weighted majority vote (medoid selection) on fuzzy/complex JSON responses.
    Finds the response r_i that maximizes the weighted sum of similarities to all other responses.
    """
    if not responses:
        return {"product_title": "", "features": []}
    if len(responses) == 1:
        return responses[0]

    best_score = -1.0
    best_response = responses[0]

    for i, r_i in enumerate(responses):
        score = 0.0
        for j, r_j in enumerate(responses):
            if i == j:
                score += weights[i]  # Similarity to itself is 1.0
            else:
                sim = compute_token_f1(r_i, r_j)
                score += weights[j] * sim

        if score > best_score:
            best_score = score
            best_response = r_i

    return best_response


def evaluate_against_gold(consensus: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """
    Evaluates the final aggregated consensus JSON against the gold standard JSON using Token F1.
    """
    return compute_token_f1(consensus, gold)


def get_beta_pdf(x: float, alpha: float, beta_val: float) -> float:
    """
    Calculates the Beta PDF value at x, clipped for numerical stability.
    """
    # Clip x to [1e-6, 1.0 - 1e-6] to avoid infinity at bounds if alpha <= 1 or beta <= 1
    x_clipped = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    return float(scipy_beta.pdf(x_clipped, alpha, beta_val))


# =====================================================================
# 2. CaMVoEstimator Class Implementation
# =====================================================================

class CaMVoEstimator:
    """
    Manages the online parameters and selection logic for the pool of LLMs.
    Uses LinUCB for contextual performance bounds and Bayesian Beta estimators
    to track labeling confidence.
    """

    def __init__(
        self,
        llm_pool: Dict[str, Dict[str, float]],
        d: int = 384,
        lambda_L: float = 1.0,
        lambda_R: float = 5.0,
        alpha_bandit: float = 0.5
    ):
        """
        Initialize the estimator with a pool of LLMs and default bandit parameters.
        
        Args:
            llm_pool: Dict of LLM configurations. E.g.:
                      {"gpt-4o": {"cost": 0.005, "prior_accuracy": 0.90}, ...}
            d: Dimension of context embeddings.
            lambda_L: LinUCB regularization (prior precision multiplier for identity).
            lambda_R: Laplace smoothing regularization strength.
            alpha_bandit: Exploration parameter for LinUCB confidence bound width.
        """
        self.llm_names = list(llm_pool.keys())
        self.num_models = len(self.llm_names)
        self.llm_costs = np.array([llm_pool[name]["cost"] for name in self.llm_names])
        self.d = d
        self.lambda_L = lambda_L
        self.lambda_R = lambda_R
        self.alpha_bandit = alpha_bandit

        # 1. Initialize LinUCB matrices: A_i = lambda_L * I_d, b_i = zeros(d)
        self.A = [self.lambda_L * np.eye(self.d) for _ in range(self.num_models)]
        self.b = [np.zeros(self.d) for _ in range(self.num_models)]

        # 2. Initialize Beta distribution parameters: alpha=1.0, beta=1.0 for each LLM
        # We track success (h=1) and failure (h=0) distributions separately
        self.alpha_1 = [1.0 for _ in range(self.num_models)]
        self.beta_1 = [1.0 for _ in range(self.num_models)]
        self.alpha_0 = [1.0 for _ in range(self.num_models)]
        self.beta_0 = [1.0 for _ in range(self.num_models)]

        # Track history of estimated q_values to compute variance and update Beta via Method of Moments
        self.history_q_success = [list() for _ in range(self.num_models)]
        self.history_q_failure = [list() for _ in range(self.num_models)]

        # 3. Track empirical accuracies & query counts
        self.historical_rewards = [list() for _ in range(self.num_models)]
        self.historical_accuracy = np.array([
            llm_pool[name].get("prior_accuracy", 0.8) for name in self.llm_names
        ])
        self.N = np.zeros(self.num_models, dtype=int)
        self.round_count = 0

    def estimate_confidence(self, e_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the expected weights and regularized Beta confidence bounds for all LLMs.
        
        Args:
            e_t: Context embedding of shape [d]
            
        Returns:
            Tuple of:
              - confidences (L): 1D array of Laplace-smoothed Beta confidences for all models.
              - weights (w): 1D array of expected correctness weights for majority voting.
        """
        # Ensure embedding is normalized
        norm = np.linalg.norm(e_t)
        if norm > 0:
            e_t = e_t / norm

        confidences = np.zeros(self.num_models)
        weights = np.zeros(self.num_models)

        for i in range(self.num_models):
            # Compute LinUCB inverse (solve for stability)
            inv_A = np.linalg.inv(self.A[i])

            # q_t(e_t) = e_t^T * A_i^-1 * b_i
            q = float(np.dot(e_t, np.dot(inv_A, self.b[i])))
            
            # C_t(e_t) = alpha * sqrt(e_t^T * A_i^-1 * e_t)
            c = self.alpha_bandit * np.sqrt(max(0.0, float(np.dot(e_t, np.dot(inv_A, e_t)))))
            
            # theta_t = clip(q - c, 0.0, 1.0)
            theta = np.clip(q - c, 0.0, 1.0)

            # Pass theta through Beta conditional likelihoods
            # success (h=1) distribution: Beta(alpha_1, beta_1)
            pdf_1 = get_beta_pdf(theta, self.alpha_1[i], self.beta_1[i])
            # failure (h=0) distribution: Beta(alpha_0, beta_0)
            pdf_0 = get_beta_pdf(theta, self.alpha_0[i], self.beta_0[i])

            # Apply Bayes' Rule: P(h=1 | theta) = mu * pdf_1 / (mu * pdf_1 + (1 - mu) * pdf_0)
            mu = self.historical_accuracy[i]
            denom = mu * pdf_1 + (1.0 - mu) * pdf_0

            if denom < 1e-12:
                # Fallback to the historical accuracy prior if denominator is near zero
                L_bar = mu
            else:
                L_bar = (mu * pdf_1) / denom

            # Apply Laplace smoothing to the LCB to prevent overconfidence in early rounds
            log_term = np.log(self.round_count + 1)
            num = self.N[i] + self.lambda_R * log_term / 2.0
            den = self.N[i] + self.lambda_R * log_term
            term = num / den if den > 0 else 0.5
            
            L = L_bar * term
            
            confidences[i] = L
            # weight w_i,t = mu_i,t-1 * q_i,t(e_t)
            weights[i] = mu * np.clip(q, 0.0, 1.0)

        return confidences, weights

    def calculate_combined_confidence(
        self,
        selected_indices: List[int],
        confidences: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Approximates the confidence of the weighted majority vote of a subset using Beta CDF.
        Formula: 1 - F_Beta(0.5; W_L, W - W_L)
        """
        W_L = sum(weights[i] * confidences[i] for i in selected_indices)
        W = sum(weights[i] for i in selected_indices)

        alpha_shape = W_L
        beta_shape = W - W_L

        # Ensure parameters are positive
        alpha_shape = max(alpha_shape, 1e-6)
        beta_shape = max(beta_shape, 1e-6)

        return float(1.0 - scipy_beta.cdf(0.5, alpha_shape, beta_shape))

    def run_oracle(
        self,
        confidences: np.ndarray,
        weights: np.ndarray,
        delta: float = 0.95,
        k_min: int = 2,
        warm_start_rounds: int = 10
    ) -> List[int]:
        """
        Greedy selection algorithm. Sorts LLMs by cost, and selects the cheapest
        subset whose combined Beta CDF confidence exceeds delta * confidence_all.
        Enforces size >= k_min.
        Queries all models during the warm_start_rounds to initialize parameters.
        """
        all_indices = list(range(self.num_models))
        
        # Force querying all models during the warm-start phase
        if self.round_count < warm_start_rounds:
            return all_indices
            
        # Calculate confidence of the full ensemble (all models)
        confidence_all = self.calculate_combined_confidence(all_indices, confidences, weights)
        
        # Target is delta * confidence_all (relative confidence constraint)
        target_confidence = delta * confidence_all
        
        # Sort LLM indices by cost ascending
        sorted_indices = np.argsort(self.llm_costs)
        
        selected_indices: List[int] = []
        for idx in sorted_indices:
            selected_indices.append(int(idx))
            if len(selected_indices) >= k_min:
                conf = self.calculate_combined_confidence(selected_indices, confidences, weights)
                if conf >= target_confidence:
                    return selected_indices

        # Fallback: query all models if target confidence cannot be met
        return all_indices

    def update_parameters(
        self,
        e_t: np.ndarray,
        selected_indices: List[int],
        rewards: Dict[int, float]
    ):
        """
        Updates LinUCB matrices, empirical accuracies, and Beta parameters for queried LLMs.
        
        Args:
            e_t: Context embedding vector.
            selected_indices: Indices of LLMs that were queried in this round.
            rewards: Dict mapping LLM index to continuous reward (Token F1).
        """
        self.round_count += 1
        e_col = e_t.reshape(-1, 1)

        for i in selected_indices:
            reward = rewards[i]
            self.N[i] += 1

            # 1. Update LinUCB parameters
            self.A[i] += np.dot(e_col, e_col.T)
            self.b[i] += reward * e_t

            # 2. Update empirical rewards and historical mean accuracy
            self.historical_rewards[i].append(reward)
            self.historical_accuracy[i] = float(np.mean(self.historical_rewards[i]))

            # 3. Update Beta shape parameters using Method of Moments
            # Compute current estimated confidence q_t(e_t) = e_t^T * A_i^-1 * b_i
            inv_A = np.linalg.inv(self.A[i])
            q_val = float(np.dot(e_t, np.dot(inv_A, self.b[i])))
            q_val = np.clip(q_val, 0.0, 1.0)

            # h=1 for success (high reward), h=0 for failure
            h = 1 if reward >= 0.5 else 0
            if h == 1:
                self.history_q_success[i].append(q_val)
                history = self.history_q_success[i]
            else:
                self.history_q_failure[i].append(q_val)
                history = self.history_q_failure[i]

            # Method of moments requires at least 2 samples to compute variance
            if len(history) >= 2:
                m = float(np.mean(history))
                s2 = float(np.var(history, ddof=1)) # sample variance

                # Numerical stability: clamp variance below theoretical maximum m*(1-m)
                max_s2 = m * (1.0 - m) - 1e-6
                s2 = np.clip(s2, 1e-6, max(1e-6, max_s2))

                if s2 < m * (1.0 - m):
                    factor = (m * (1.0 - m) / s2) - 1.0
                    alpha_est = m * factor
                    beta_est = (1.0 - m) * factor

                    # Clip estimated shape parameters to prevent overflow or underflow
                    alpha_est = float(np.clip(alpha_est, 1e-3, 1000.0))
                    beta_est = float(np.clip(beta_est, 1e-3, 1000.0))

                    if h == 1:
                        self.alpha_1[i] = alpha_est
                        self.beta_1[i] = beta_est
                    else:
                        self.alpha_0[i] = alpha_est
                        self.beta_0[i] = beta_est


# =====================================================================
# 3. Main Processing Loop
# =====================================================================

def run_annotation_loop(
    product_texts: List[str],
    embeddings: np.ndarray,
    estimator: CaMVoEstimator,
    query_fn: Callable[[str, np.ndarray, str, Dict[str, Any]], Dict[str, Any]],
    delta: float = 0.95,
    k_min: int = 2
) -> Dict[str, Any]:
    """
    Iterates through the dataset sequentially, selecting LLMs, aggregating votes,
    and updating estimator parameters.
    
    Args:
        product_texts: List of unstructured text strings.
        embeddings: Numpy array of context embeddings of shape [N, d].
        estimator: CaMVoEstimator instance managing state.
        query_fn: Callback function to simulate querying an LLM: query_fn(llm_name, embedding, text, gold_json) -> response_dict.
        delta: Confidence threshold parameter.
        k_min: Minimum number of LLMs to select.
    """
    total_cost = 0.0
    total_consensus_accuracy = 0.0
    queries_count = np.zeros(estimator.num_models, dtype=int)
    consensus_results: List[Dict[str, Any]] = []

    print(f"\nStarting CaMVo loop over {len(product_texts)} instances...")
    print(f"Target Confidence Threshold delta = {delta}, Min LLMs k_min = {k_min}")
    print("-" * 80)

    for t in range(len(product_texts)):
        text = product_texts[t]
        e_t = embeddings[t]

        # 1. Estimate confidence and weights
        confidences, weights = estimator.estimate_confidence(e_t)

        # 2. Run greedy selection oracle
        selected_indices = estimator.run_oracle(confidences, weights, delta=delta, k_min=k_min)

        # Calculate cost for this query
        round_cost = sum(estimator.llm_costs[idx] for idx in selected_indices)
        total_cost += round_cost
        for idx in selected_indices:
            queries_count[idx] += 1

        # 3. Simulate/Query selected LLMs
        gold_json = {
            "product_title": text[:40],
            "features": [f"Feature extracted from title snippet: {text[:20]}"]
        }

        responses: List[Dict[str, Any]] = []
        rewards: Dict[int, float] = {}

        for idx in selected_indices:
            name = estimator.llm_names[idx]
            pred = query_fn(name, e_t, text, gold_json)
            responses.append(pred)

        # 4. Aggregate results using fuzzy majority vote
        selected_weights = [weights[idx] for idx in selected_indices]
        consensus = fuzzy_majority_vote(responses, selected_weights)
        consensus_results.append(consensus)

        # Calculate Token F1 against consensus for reward
        for idx, pred in zip(selected_indices, responses):
            rewards[idx] = compute_token_f1(pred, consensus)

        # Calculate accuracy against true gold
        accuracy = evaluate_against_gold(consensus, gold_json)
        total_consensus_accuracy += accuracy

        # 5. Parameter updates (only if queried subset size > 1)
        if len(selected_indices) > 1:
            estimator.update_parameters(e_t, selected_indices, rewards)

        # Periodic logging
        if (t + 1) % 20 == 0 or t == len(product_texts) - 1:
            avg_cost = total_cost / (t + 1)
            avg_acc = total_consensus_accuracy / (t + 1)
            selected_names = [estimator.llm_names[i] for i in selected_indices]
            print(f"Round {t+1:03d} | Selected: {selected_names} | Consensus Acc: {avg_acc:.3f} | Avg Cost: {avg_cost:.5f}")

    print("=" * 80)
    print("CaMVo loop completed!")
    print(f"Final Average Consensus Accuracy: {total_consensus_accuracy / len(product_texts):.4f}")
    print(f"Final Total API cost: {total_cost:.5f}")
    print("Model Query Statistics:")
    for i, name in enumerate(estimator.llm_names):
        rate = queries_count[i] / len(product_texts)
        print(f"  - {name:16s}: {queries_count[i]:03d} queries ({rate:.1%}) | Estimated Accuracy: {estimator.historical_accuracy[i]:.3f}")

    return {
        "total_cost": total_cost,
        "average_accuracy": total_consensus_accuracy / len(product_texts),
        "queries_count": queries_count,
        "consensus_results": consensus_results
    }
