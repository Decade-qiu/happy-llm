import json
import os
import logging
import random
from typing import Generator
from transformers import AutoTokenizer

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
    normalizers,
)

import psutil, sys, time, threading

def start_memory_guard(
    interval_sec: float = 1.0,
    max_memory_gb: float = 3.0
):
    """
    Start a memory monitoring thread that will terminate the process if memory usage exceeds max_memory_gb.
    """
    process = psutil.Process(os.getpid())

    def monitor():
        max_bytes = max_memory_gb * 1024 * 1024 * 1024
        logging.info(f"[MemoryGuard] Start! (Threshold: {max_memory_gb} GB)")

        while True:
            rss = process.memory_info().rss
            if rss > max_bytes:
                logging.error(
                    f"[MemoryGuard] Warning！Current {rss/1024/1024:.2f} MB > {max_memory_gb} GB\n"
                )
                os._exit(1)
            time.sleep(interval_sec)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

# --- Use the logging module for informative output ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """
    Safely reads text content from a JSONL file, yielding one text string at a time.

    Args:
        file_path (str): The path to the JSONL file.

    Yields:
        Generator[str, None, None]: A generator that yields the value of the 'text' field from each valid JSON line.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' in data and isinstance(data['text'], str):
                    yield data['text']
                else:
                    logging.warning(f"Line {line_num}: 'text' field is missing or not a string.")
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON in line {line_num}: {line.strip()}")
                continue

def create_tokenizer_config(save_dir: str) -> None:
    """
    Creates the necessary configuration files (tokenizer_config.json, special_tokens_map.json)
    for the tokenizer to be loaded by Hugging Face's AutoTokenizer.

    Args:
        save_dir (str): The directory where the configuration files will be saved.
    """
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
    }
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192):
    """
    Trains a robust Byte-Level BPE tokenizer from text data and saves it.

    Args:
        data_path (str): Path to the training data file (JSONL format).
        save_dir (str): Directory where the trained tokenizer will be saved.
        vocab_size (int): The target size of the vocabulary.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define special tokens. Their order here determines their assigned IDs (0, 1, 2, ...).
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # Initialize a BPE tokenizer model.
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Configure a safer, lossless Normalizer.
    tokenizer.normalizer = normalizers.NFC() # type: ignore

    # Configure Byte-Level pre-tokenization and decoding.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # type: ignore
    tokenizer.decoder = decoders.ByteLevel() # type: ignore

    # Configure the trainer.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Correctly calculate the iterator length for the progress bar.
    logging.info("Calculating the total number of samples...")
    try:
        # This is a memory-efficient way to count lines in a large file.
        num_texts = sum(1 for _ in open(data_path, 'r', encoding='utf-8'))
    except FileNotFoundError:
        logging.error(f"Data file not found at: {data_path}")
        return
        
    logging.info(f"Found {num_texts} samples for training.")

    # Create the iterator for training.
    texts_iterator = read_texts_from_jsonl(data_path)
    
    logging.info(f"Starting tokenizer training with data from {data_path}")
    # Train the model using the iterator.
    tokenizer.train_from_iterator(
        texts_iterator, 
        trainer=trainer, 
        length=num_texts  # Use the correct number of samples.
    )

    # Save the core tokenizer file and the configuration files.
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    create_tokenizer_config(save_dir)
    logging.info(f"Tokenizer training complete. Files saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str):
    """evaluate the trained tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "你最近过得还好吗？"},
        {"role": "assistant", "content": "谢谢关心，我很好，你呢？"},
        {"role": "user", "content": "我也很好！"},
        {"role": "assistant", "content": "那太好了！有什么我可以帮你的吗？"}
    ]
    
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=512)
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"Encoded IDs: {encoded['input_ids']}")
    print(f"Decoded text: {decoded}")
    print("Decoded text matches original:", decoded == prompt)

    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\n你是谁啊?<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)
    

if __name__ == '__main__':
    random.seed(1773771)

    start_memory_guard(
        interval_sec=10,
        max_memory_gb=300
    )

    data_path = "../data/pretrain_data.jsonl"
    save_dir = "./"

    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=8096
    )

    eval_tokenizer(save_dir)

    # eval_tokenizer("/Users/qzj/Desktop/Development/happy-llm/docs/chapter5/code/tokenizer_k/")