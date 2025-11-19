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

        self.offsets = [0]
        self.file_handle = None
        
        print(f"Building index for {data_path} ...")

        with open(data_path, "rb") as f:
            # index = 0
            while True:
                line = f.readline()
                if not line:
                    break
                self.offsets.append(f.tell())
                index += 1
                if index == 1024 * 1024:
                    break 

        if self.offsets[-1] == self.offsets[-2]:
             self.offsets.pop()
        
        self.total_samples = len(self.offsets) - 1
        print(f"Index built. Found {self.total_samples} samples.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index: int):
        if self.file_handle is None:
            self.file_handle = open(self.data_path, "rb")

        start_pos = self.offsets[index]
        end_pos = self.offsets[index + 1]
        length = end_pos - start_pos

        self.file_handle.seek(start_pos)
        line_bytes = self.file_handle.read(length)

        try:
            sample = json.loads(line_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            print(f"Error decoding line at index {index}")
            return self.__getitem__((index + 1) % self.total_samples)

        input_ids = sample['tokens']
        doc_boundaries = sample['doc_boundaries']
        
        attention_mask = self._create_attention_mask(doc_boundaries, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool)
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to combine a list of samples into a batch.
        """
        # [batch_size, seq_length]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        # [batch_size, 1, seq_length, seq_length]
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        attention_mask.unsqueeze_(1)  

        return {
            'input_ids': input_ids,           # [batch_size, seq_length]
            'attention_mask': attention_mask, # [batch_size, seq_length, seq_length]
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
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)

        for start, end in doc_boundaries:
            # Create a causal mask for the segment of this document
            doc_len = end - start
            causal_mask = torch.tril(torch.ones(doc_len, doc_len, dtype=torch.bool))
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