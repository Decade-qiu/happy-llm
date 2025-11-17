import os
import json
from tqdm import tqdm


def process_pretrain_data(src, dst):
    def split_text(text, chunk_size=512):
        """chunk the text into pieces of chunk_size"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    with open(dst, 'w', encoding='utf-8') as pretrain:
        with open(src, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data, desc=f"Processing", leave=False):
                line = json.loads(line)
                text = line['text']
                chunks = split_text(text)
                for chunk in chunks:
                    pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')



def process_finetune_data(src, dst):
    def convert_message(data):
        """convert the conversation data to the format required by the model"""
        message = [
            {"role": "system", "content": "你是一个AI助手"},
        ]
        for item in data:
            if item['from'] == 'human':
                message.append({'role': 'user', 'content': item['value']})
            elif item['from'] == 'assistant':
                message.append({'role': 'assistant', 'content': item['value']})
        return message

    with open(dst, 'a', encoding='utf-8') as sft:
        with open(src, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for item in tqdm(data, desc="Processing", unit="lines"):
                item = json.loads(item)
                message = convert_message(item['conversations'])
                sft.write(json.dumps(message, ensure_ascii=False) + '\n')

                

if __name__ == "__main__":
    # process pretrain data
    process_pretrain_data(
        src='./mobvoi_seq_monkey_general_open_corpus.jsonl',
        dst='./pretrain_data.jsonl'
    )

    # process finetune data
    process_finetune_data(
        src='./train_3.5M_CN.json',
        dst='./finetune_data.jsonl'
    )