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
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to combine a list of samples into a batch.
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])

        return {
            'input_ids': input_ids,           # [batch_size, seq_length]
            'attention_mask': attention_mask, # [batch_size, seq_length, seq_length]
        }
    
    def _prepare_samples(self):
        """
        Optimized method to prepare samples by processing documents in a stream.
        This avoids loading the entire dataset into memory.
        """
        num_documents = sum(1 for _ in open(self.data_path, 'r'))

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


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']  # <|im_start|>assistant\n
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0
        
        while i <= n - a_length:
            # 检查当前位置是否匹配目标子序列
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 从子序列结束的位置开始查找第一个 4 (eos_token_id)
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 结束位置设为j（包含4）
                    # 标记区间为1（包括start到end）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 跳过当前子序列，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)