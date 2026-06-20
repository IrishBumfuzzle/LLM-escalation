#!/usr/bin/env python3
import os
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Generate Amazon product title embeddings using nomic-embed-text-v1.5 with GPU acceleration and resumability."
    )
    parser.add_argument("--input-csv", type=str, default="products.csv", help="Path to input products CSV file.")
    parser.add_argument("--output-csv", type=str, default="products_with_embeddings.csv", help="Path to save final products CSV file.")
    parser.add_argument("--progress-csv", type=str, default="products_progress.csv", help="Path to progress tracking file.")
    parser.add_argument("--column", type=str, default="title", help="Column name to generate embeddings from.")
    parser.add_argument("--batch-size", type=str, default="128", help="Batch size for embedding generation. Default 128.")
    parser.add_argument("--dimension", type=int, default=768, help="Target embedding dimension. Support Matryoshka truncation (e.g. 384 or 768). Default 768.")
    parser.add_argument("--prefix", type=str, default="search_document: ", help="Prefix to prepend to titles. Default 'search_document: '.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (for testing).")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu'). Auto-detects cuda by default.")
    parser.add_argument("--force-restart", action="store_true", help="Delete progress file and start embedding generation from scratch.")
    
    args = parser.parse_args()

    # Parse batch size to int
    try:
        batch_size = int(args.batch_size)
    except ValueError:
        print(f"Error: Invalid batch size '{args.batch_size}'", file=sys.stderr)
        sys.exit(1)

    # 1. Setup device
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Check source file
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading input dataset from '{args.input_csv}'...")
    df = pd.read_csv(args.input_csv)
    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows}")

    if args.limit:
        print(f"Testing mode: Limiting processing to first {args.limit} rows.")
        df = df.iloc[:args.limit]

    # Verify unique identifiers
    if 'parent_asin' not in df.columns:
        print("Error: 'parent_asin' column is missing from input CSV.", file=sys.stderr)
        sys.exit(1)

    if args.column not in df.columns:
        print(f"Error: Column to embed '{args.column}' is missing from input CSV.", file=sys.stderr)
        sys.exit(1)

    # 3. Handle progress file & resumability
    progress_exists = os.path.exists(args.progress_csv)
    if args.force_restart and progress_exists:
        print(f"Force restart: deleting existing progress file '{args.progress_csv}'...")
        os.remove(args.progress_csv)
        progress_exists = False

    processed_asins = {}
    if progress_exists:
        print(f"Reading existing progress from '{args.progress_csv}'...")
        try:
            # Load processed rows
            progress_df = pd.read_csv(args.progress_csv)
            for idx, row in progress_df.iterrows():
                asin = row['parent_asin']
                # Store serialized representation directly
                processed_asins[asin] = row['embedding']
            print(f"Found {len(processed_asins)} completed embeddings in progress file.")
        except Exception as e:
            print(f"Warning: Failed to parse progress file: {e}. Starting fresh.", file=sys.stderr)
            processed_asins = {}

    # Filter rows that need processing
    to_process_df = df[~df['parent_asin'].isin(processed_asins)]
    rows_to_process = len(to_process_df)
    
    if rows_to_process > 0:
        print(f"Generating embeddings for {rows_to_process} remaining rows...")
        
        # Initialize SentenceTransformers model
        print("Loading nomic-embed-text-v1.5 model...")
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
            model = model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)

        # Open progress file in append mode
        # If it didn't exist or was deleted, write header first
        write_header = not os.path.exists(args.progress_csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.progress_csv)) if os.path.dirname(args.progress_csv) else '.', exist_ok=True)
        
        # We process in python loop to append incrementally
        # Chunk the rows to process
        indices = to_process_df.index.tolist()
        
        with open(args.progress_csv, mode='a', encoding='utf-8', newline='') as pf:
            writer = csv.writer(pf)
            if write_header:
                writer.writerow(['parent_asin', 'embedding'])
                pf.flush()

            pbar = tqdm(total=rows_to_process, desc="Computing embeddings")
            for i in range(0, rows_to_process, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_df = to_process_df.loc[batch_indices]
                
                # Extracted text with task prefix
                texts = [f"{args.prefix}{str(val)}" for val in batch_df[args.column].fillna("")]
                asins = batch_df['parent_asin'].tolist()

                try:
                    # Generate embeddings (shape: [batch_size, 768])
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )

                    # Handle Matryoshka Representation Learning dimension truncation & L2-normalization
                    if args.dimension and args.dimension != 768:
                        embeddings = embeddings[:, :args.dimension]
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1.0, norms)
                        embeddings = embeddings / norms

                    # Write to progress file batch-by-batch
                    for asin, emb in zip(asins, embeddings):
                        # Serialize floats to 6 decimal places to conserve file size
                        serialized_emb = json.dumps([round(float(x), 6) for x in emb])
                        writer.writerow([asin, serialized_emb])
                        processed_asins[asin] = serialized_emb
                    
                    pf.flush() # Ensure it's written to disk immediately
                    pbar.update(len(batch_indices))

                except torch.cuda.OutOfMemoryError:
                    print("\nCUDA Out Of Memory! Try running with a smaller batch size (e.g. --batch-size 32).", file=sys.stderr)
                    sys.exit(1)
                except Exception as e:
                    print(f"\nError processing batch: {e}", file=sys.stderr)
                    sys.exit(1)

            pbar.close()
    else:
        print("All rows are already processed!")

    # 4. Final Compile: Merge progress file with original columns
    print("Compiling final output dataset...")
    # Map ASIN to the generated embedding string
    df['embedding'] = df['parent_asin'].map(processed_asins).fillna(df['embedding'])

    print(f"Writing final dataset to '{args.output_csv}'...")
    df.to_csv(args.output_csv, index=False)
    print("Embedding generation completed successfully!")

    # Prompt about the progress file
    print(f"You can now safely delete the progress file '{args.progress_csv}' if desired.")

if __name__ == "__main__":
    main()
