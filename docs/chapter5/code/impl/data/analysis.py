import json
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Setup logging for clear output ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_statistics(lengths: list, name: str):
    """Calculates and prints key statistics for a list of lengths."""
    if not lengths:
        logging.warning(f"No valid data found for {name} statistics.")
        return

    lengths_np = np.array(lengths)
    
    stats = {
        "Count": len(lengths_np),
        "Average": np.mean(lengths_np),
        "Min": np.min(lengths_np),
        "Max": np.max(lengths_np),
        "Median (50th)": np.median(lengths_np),
        "90th Percentile": np.percentile(lengths_np, 90),
        "95th Percentile": np.percentile(lengths_np, 95),
        "99th Percentile": np.percentile(lengths_np, 99),
    }

    logging.info(f"--- {name} Length Statistics ---")
    for key, value in stats.items():
        logging.info(f"{key:<15}: {value:,.2f}")
    logging.info("-" * (20 + len(name)))


def plot_distribution(lengths: list, title: str, xlabel: str, output_filename: str):
    """Generates and saves a histogram of the length distribution."""
    if not lengths:
        logging.warning(f"Cannot plot distribution for '{title}', no data.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(lengths, bins=100, kde=False)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Clip the x-axis to the 99.5th percentile for better visualization of the main distribution
    # This prevents extreme outliers from squishing the plot.
    upper_limit = np.percentile(lengths, 99.5)
    plt.xlim(0, upper_limit)
    
    # Use a log scale for the y-axis to better see less frequent lengths
    plt.yscale('log')
    
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=150)
    logging.info(f"Distribution plot saved to: {output_filename}")
    plt.close()


def analyze_dataset_lengths(src_path: str, tokenizer_path: str):
    """
    Analyzes a JSONL dataset for character and token lengths and generates reports.

    Args:
        src_path (str): Path to the source JSONL file.
        tokenizer_path (str): Path to the pre-trained Hugging Face tokenizer directory.
    """
    logging.info("Starting dataset analysis...")
    
    # --- Load Tokenizer ---
    try:
        logging.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer from '{tokenizer_path}'. Error: {e}")
        return

    char_lengths = []
    token_lengths = []

    # --- First, get total line count for an accurate progress bar ---
    try:
        logging.info("Counting total lines in the source file for progress tracking...")
        with open(src_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        logging.info(f"Found {num_lines:,} lines to process.")
    except FileNotFoundError:
        logging.error(f"Source file not found at: {src_path}")
        return

    # --- Process the file line-by-line ---
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="Analyzing Documents"):
            try:
                data = json.loads(line)
                text = data.get('text')

                if text and isinstance(text, str):
                    # 1. Character-level analysis
                    char_lengths.append(len(text))

                    # 2. Token-level analysis
                    # We use add_special_tokens=False to analyze the content length itself,
                    # without adding extra BOS/EOS tokens which would skew the stats.
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                    token_lengths.append(len(token_ids))

            except (json.JSONDecodeError, KeyError):
                # Silently skip malformed lines, as they are not part of the valid dataset
                continue
    
    logging.info("Analysis complete. Generating reports...")

    # --- Generate Statistics and Plots ---
    # Character statistics
    print_statistics(char_lengths, "Character")
    plot_distribution(
        char_lengths, 
        "Distribution of Document Character Lengths", 
        "Length (Characters)",
        "character_length_distribution.png"
    )

    # Token statistics
    print_statistics(token_lengths, "Token")
    plot_distribution(
        token_lengths,
        "Distribution of Document Token Lengths",
        "Length (Tokens)",
        "token_length_distribution.png"
    )


if __name__ == "__main__":
    # ==============================================================================
    # --- CONFIGURATION: PLEASE UPDATE THESE PATHS ---
    # ==============================================================================
    
    # IMPORTANT: Set this to the path of your 35GB JSONL file.
    # For demonstration, I'm creating a dummy file.
    SRC_FILE = "dummy_pretrain_data.jsonl" 
    
    # IMPORTANT: Set this to the path where you saved your trained tokenizer.
    # For demonstration, I'm using a pre-trained model from Hugging Face.
    TOKENIZER_PATH = "my_trained_tokenizer" # Or for example: "gpt2"

    # --- Run the analysis ---
    analyze_dataset_lengths(src_path=SRC_FILE, tokenizer_path=TOKENIZER_PATH)