import json
import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast



class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id

        self.data_path = data_path
        
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """
        Optimized method to prepare samples by processing documents in a stream.
        This avoids loading the entire dataset into memory.
        """
        num_documents = sum(1 for line in open(self.data_path, 'r'))

        buffer_tokens = []
        buffer_boundaries = []
        total_tokens_processed = 0

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_documents, desc="Processing Documents"):
                # 1. Load and tokenize one document
                sample_text = json.loads(line)['text']
                doc_tokens = self.tokenizer.encode(sample_text, add_special_tokens=False)
                doc_tokens.append(self.eos_token_id)
                total_tokens_processed += len(doc_tokens)
                
                # 2. Add the new document's tokens and its boundary to the buffer
                new_doc_start = len(buffer_tokens)
                buffer_tokens.extend(doc_tokens)
                buffer_boundaries.append((new_doc_start, len(buffer_tokens)))

                # 3. Create as many full samples as possible from the current buffer
                while len(buffer_tokens) >= self.max_length:
                    # Take max_length tokens for the new sample
                    sample_tokens = buffer_tokens[:self.max_length]
                    
                    # Calculate document boundaries for this specific sample
                    sample_doc_boundaries = []
                    for doc_start, doc_end in buffer_boundaries:
                        # Calculate the overlap between the document [doc_start, doc_end)
                        # and the sample's range [0, self.max_length)
                        overlap_start = max(0, doc_start)
                        overlap_end = min(self.max_length, doc_end)
                        
                        if overlap_start < overlap_end:
                            sample_doc_boundaries.append((overlap_start, overlap_end))

                    # Add the finalized sample
                    self.samples.append({
                        'tokens': sample_tokens,
                        'doc_boundaries': sample_doc_boundaries
                    })
                    
                    # 4. Update the buffers for the next iteration
                    # Remove the tokens that were just used
                    buffer_tokens = buffer_tokens[self.max_length:]
                    
                    # Update boundaries: shift them left and remove those no longer in the buffer
                    new_boundaries = []
                    for doc_start, doc_end in buffer_boundaries:
                        new_start = doc_start - self.max_length
                        new_end = doc_end - self.max_length
                        if new_end > 0: # Keep the boundary only if it's still relevant
                            new_boundaries.append((new_start, new_end))
                    
                    buffer_boundaries = new_boundaries

        num_samples = len(self.samples)
        print(f"Prepared {num_samples} samples from {num_documents} documents")
        print(f"Total tokens: {total_tokens_processed}, Utilized: {num_samples * self.max_length}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        input_ids = sample['tokens']
        doc_boundaries = sample['doc_boundaries']
        
        # intra-document attention mask
        attention_mask = self._create_attention_mask(doc_boundaries, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def _create_attention_mask(self, doc_boundaries, seq_length):
        """
        Create an attention mask where tokens can only attend to tokens within the same document.
        Args:
            doc_boundaries (List[Tuple[int, int]]): List of (start, end) indices for each document in the sample.
            seq_length (int): Total sequence length.
        Returns:
            torch.Tensor: Attention mask of shape (seq_length, seq_length).
        1 indicates attention is allowed, 0 indicates no attention.
        """
        mask = torch.zeros(seq_length, seq_length, dtype=torch.long)

        for start, end in doc_boundaries:
            # Create a causal mask for the segment of this document
            doc_len = end - start
            causal_mask = torch.tril(torch.ones(doc_len, doc_len, dtype=torch.long))
            # Place this causal mask in the correct block of the main attention mask
            mask[start:end, start:end] = causal_mask
        
        return mask

# --- 测试部分 ---

# 1. 模拟 Tokenizer
class MockTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.next_id = 1
        self.eos_token_id = 999  # 使用一个特殊ID表示文档结束

    def encode(self, text, add_special_tokens=False):
        tokens = []
        for word in text.split():
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.next_id += 1
            tokens.append(self.word_to_id[word])
        return tokens

def create_dummy_data_file(file_path="test_data.jsonl"):
    """生成一个临时的测试数据文件"""
    documents = [
        "This is the first document.",      # 5 words
        "This is a shorter second one.",    # 6 words
        "The third document is a bit longer than the first.", # 10 words
        "And a final one."                  # 4 words
    ]
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc_text in documents:
            f.write(json.dumps({'text': doc_text}) + '\n')
    print(f"Created dummy data file at: {file_path}")

def run_tests():
    """执行所有测试"""
    data_path = "test_data.jsonl"
    create_dummy_data_file(data_path)
    
    max_length = 10  # 使用一个较小的 max_length 以方便调试
    tokenizer = MockTokenizer()
    
    print("\n--- Initializing Dataset ---")
    dataset = PretrainDataset(data_path, tokenizer, max_length=max_length)
    
    print("\n--- Running Assertions ---")
    
    # 测试1: 检查样本数量
    # Doc1 (5) + EOS (1) = 6
    # Doc2 (6) + EOS (1) = 7
    # Doc3 (10) + EOS (1) = 11
    # Doc4 (4) + EOS (1) = 5
    # Total tokens = 6 + 7 + 11 + 5 = 29
    # Expected samples = 29 // 10 = 2
    assert len(dataset) == 2, f"Expected 2 samples, but got {len(dataset)}"
    print("✅ Test 1: Correct number of samples generated.")

    # 测试2: 检查第一个样本的内容
    sample_0 = dataset[0]
    input_ids_0 = sample_0['input_ids']
    attn_mask_0 = sample_0['attention_mask']

    assert input_ids_0.shape == (max_length,), f"Sample 0 input_ids shape is wrong: {input_ids_0.shape}"
    assert attn_mask_0.shape == (max_length, max_length), f"Sample 0 attention_mask shape is wrong: {attn_mask_0.shape}"
    print("✅ Test 2: Sample 0 has correct tensor shapes.")

    print("\n--- Detailed Content Checks ---")
    print("Input IDs:", input_ids_0)
    print("Attention Mask:\n", attn_mask_0)

    # 手动验证第一个样本的边界和内容
    # Doc1 tokens: [1, 2, 3, 4, 5, 999] (len=6)
    # Doc2 tokens: [1, 2, 6, 7, 8, 9, 999] (len=7, new words start from id 6)
    # Concatenated: [1, 2, 3, 4, 5, 999, 1, 2, 6, 7, 8, 9, 999, ...]
    # Sample 0 should be the first 10 tokens: [1, 2, 3, 4, 5, 999, 1, 2, 6, 7]
    
    # Doc1 占据了 [0, 6)
    # Doc2 占据了 [6, 10)
    expected_boundaries_0 = [(0, 6), (6, 10)]
    assert dataset.samples[0]['doc_boundaries'] == expected_boundaries_0, \
        f"Sample 0 boundaries incorrect. Got {dataset.samples[0]['doc_boundaries']}, expected {expected_boundaries_0}"
    print("✅ Test 3: Sample 0 has correct document boundaries.")
    
    # 测试3: 检查 Attention Mask
    # 掩码应该是一个块对角矩阵，每个块都是一个下三角阵
    # 块1: 位置 [0:6, 0:6] 应该是下三角阵
    # 块2: 位置 [6:10, 6:10] 应该是下三角阵
    # 其他位置应该都是 0
    
    # 检查块1
    block1 = attn_mask_0[0:6, 0:6]
    assert torch.all(block1 == torch.tril(torch.ones(6, 6))), "Attention mask block 1 is incorrect."
    
    # 检查块2
    block2 = attn_mask_0[6:10, 6:10]
    assert torch.all(block2 == torch.tril(torch.ones(4, 4))), "Attention mask block 2 is incorrect."
    
    # 检查块外区域是否为0
    assert attn_mask_0[0:6, 6:10].sum() == 0, "Attention should not cross from doc1 to doc2."
    assert attn_mask_0[6:10, 0:6].sum() == 0, "Attention should not cross from doc2 to doc1."
    print("✅ Test 4: Sample 0 attention mask is correctly structured.")

    # 测试4: 检查第二个样本
    sample_1 = dataset[1]
    # Remaining from doc2: [8, 9, 999] (len=3)
    # Doc3 tokens: [10, 11, 12, 2, 8, 13, 14, 3, 4, 999] (len=11, some tokens are reused)
    # Buffer starts with: [8, 9, 999] + [10, 11, 12, 2, 8, 13, 14, 3, 4, 999]
    # Sample 1 should be: [8, 9, 999, 10, 11, 12, 2, 8, 13, 14]
    
    # 剩余的 Doc2 占据 [0, 3)
    # Doc3 的一部分占据 [3, 10)
    expected_boundaries_1 = [(0, 3), (3, 10)]
    assert dataset.samples[1]['doc_boundaries'] == expected_boundaries_1, \
        f"Sample 1 boundaries incorrect. Got {dataset.samples[1]['doc_boundaries']}, expected {expected_boundaries_1}"
    print("✅ Test 5: Sample 1 has correct document boundaries.")

    print(sample_1)

    print("\n--- All tests passed successfully! ---")

    # 5. 清理临时文件
    os.remove(data_path)
    print(f"Cleaned up and removed {data_path}")

if __name__ == "__main__":
    run_tests()