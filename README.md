# Amazon Product Title Embeddings with nomic-embed-text-v1.5

This directory contains the pipeline and output dataset for generating high-quality text embeddings from Amazon product listings.

## Dataset & Embedding Specifications

| Metric / Parameter | Value / Details | Description |
| :--- | :--- | :--- |
| **Model** | `nomic-ai/nomic-embed-text-v1.5` | High-performance open-weights text embedding model with support for long context (2048 tokens). |
| **Embedding Column** | `embedding` | Generated from the `title` column of the dataset. |
| **Task Prefix** | `search_document: ` | Prepended to each title as recommended by Nomic for document search indexing. |
| **Default Dimension** | **768** | Default dimension of the generated embeddings. |
| **Alternative Dimension** | **384** | Supported via Matryoshka Representation Learning (MRL) truncation with L2-normalization. |
| **Unique Identifier** | `parent_asin` | Primay key verified to have **0 duplicates** across all 88,046 rows. |

---

## File Sizes & Datasets

- **`products.csv`** (Source): **88.7 MB** (88,046 rows, 11 columns, with all-zero placeholder embeddings of length 384).
- **`products_with_embeddings.csv`** (Output): **689 MB** (88,046 rows, 11 columns, containing fully populated **768-dimensional** embeddings).

---

## Performance & Throughput

The embeddings were generated using PyTorch and SentenceTransformers with GPU acceleration on a local laptop:

- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 13.3)
- **Batch Size**: 128
- **Total Rows**: 88,046
- **Processing Speed**: **~203 rows/second**
- **Total Duration**: **7 minutes 14 seconds** (for the full run)

---

## How to Run & Configure the Pipeline

The [generate_embeddings.py](generate_embeddings.py) script is fully configurable and supports resumability out of the box. 

### Prerequisites

Create a virtual environment and install the required dependencies:
```bash
virtualenv venv
./venv/bin/pip install torch sentence-transformers pandas tqdm accelerate einops
```

### Basic Run (Default 768 dimensions)
```bash
./venv/bin/python generate_embeddings.py --input-csv products.csv --output-csv products_with_embeddings.csv
```

### Matryoshka Truncation (384 dimensions)
To match the original 384 dimensions schema of the placeholder embeddings (applying proper L2-normalization to the truncated vectors):
```bash
./venv/bin/python generate_embeddings.py --dimension 384 --output-csv products_with_embeddings_384.csv
```

### Script CLI Options

- `--input-csv` (str): Input CSV file containing titles. Default: `products.csv`.
- `--output-csv` (str): Output CSV file path. Default: `products_with_embeddings.csv`.
- `--progress-csv` (str): Output CSV tracking progress. Default: `products_progress.csv`.
- `--column` (str): Column name to embed. Default: `title`.
- `--batch-size` (int): Batch size for inference. Default: `128`.
- `--dimension` (int): Target dimension size (`768` or `384`). Default: `768`.
- `--prefix` (str): Prefix required by nomic model. Default: `search_document: `.
- `--limit` (int): Process only the first N rows (great for testing).
- `--device` (str): Specific device to use (`cuda` or `cpu`). Default: Auto-detects.
- `--force-restart`: Ignores existing progress and restarts generation from scratch.

---

## Cost-aware Majority Voting (CaMVo) Pipeline

We have implemented the online CaMVo annotation loop in `camvo.py` and a separate simulation execution script in `run_simulation.py`.

### Code Implementation Details

1. **`camvo.py`**:
   - **`CaMVoEstimator`**: An object-oriented class that manages the online state of LLMs (GPT-4o, Claude-3-Haiku, Gemini-Flash, LLaMA-3).
     - Maintains LinUCB matrices $A_i$ and vectors $b_i$ for contextual correctness probability estimation.
     - Maintains Beta distribution parameters ($\alpha_{i,h}$, $\beta_{i,h}$) for success and failure likelihoods, updated dynamically using the Method of Moments.
   - **Confidence Estimation (`estimate_confidence`)**: Computes the Lower Confidence Bound (LCB) $\theta_{i,t}$ using LinUCB, passes it through a Laplace-smoothed Bayesian Beta posterior to find the confidence $L_{i,t}$, and computes expected correctness weights $\omega_{i,t}$ for majority voting.
   - **Greedy Oracle Selection (`run_oracle`)**: Sorts LLMs by cost and dynamically finds the cheapest subset of LLMs satisfying:
     $$\text{Confidence}(\mathcal{A}) \ge \delta \cdot \text{Confidence}(\mathcal{ALL})$$
     Enforcing a minimum of $k_{min}$ models. Uses a Beta CDF approximation to calculate subset confidence.
   - **Main Annotation Loop (`run_annotation_loop`)**: Iterates through titles and embeddings, fetches LCBs, queries selected LLMs via a callback, aggregates results using `fuzzy_majority_vote` (medoid similarity), calculates Token F1 rewards, and updates parameters.

2. **`run_simulation.py`**:
   - Implements simulation setup, generating synthetic product titles/embeddings or reading the real generated embeddings CSV file (`products_with_embeddings.csv`).
   - Defines a mock LLM querying function that simulates model accuracy based on embedding features (contextual routing).
   - Instantiates `CaMVoEstimator` and executes `run_annotation_loop`.

### Running the CaMVo Simulation

To execute the simulation:
```bash
./venv/bin/python run_simulation.py
```

If the generated `products_with_embeddings.csv` exists in the workspace, it will automatically load the real listings and 768-dimensional embeddings to run the online loop. Otherwise, it will automatically generate a synthetic dataset to run the simulation.

