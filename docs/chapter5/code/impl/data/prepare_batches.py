import json
import argparse
from tqdm import tqdm


def prepare_samples(input_path, output_path, tokenizer, max_length=1024):
    eos_token_id = tokenizer.eos_token_id

    num_documents = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))

    buffer_tokens = []
    buffer_boundaries = []
    total_tokens_processed = 0
    out_f = open(output_path, "w", encoding="utf-8")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_documents, desc="Processing Documents"):
            sample_text = json.loads(line)['text']

            # tokenize document
            doc_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
            doc_tokens.append(eos_token_id)
            total_tokens_processed += len(doc_tokens)

            new_doc_start = len(buffer_tokens)
            buffer_tokens.extend(doc_tokens)
            buffer_boundaries.append((new_doc_start, len(buffer_tokens)))

            # create samples
            while len(buffer_tokens) >= max_length:
                sample_tokens = buffer_tokens[:max_length]

                # calculate boundaries for this sample
                sample_doc_boundaries = []
                for doc_start, doc_end in buffer_boundaries:
                    overlap_start = max(0, doc_start)
                    overlap_end = min(max_length, doc_end)

                    if overlap_start < overlap_end:
                        sample_doc_boundaries.append(
                            (overlap_start, overlap_end)
                        )

                # write out
                out_f.write(json.dumps({
                    "tokens": sample_tokens,
                    "doc_boundaries": sample_doc_boundaries
                }, ensure_ascii=False) + "\n")

                # remove used tokens
                buffer_tokens = buffer_tokens[max_length:]

                # update boundaries
                new_boundaries = []
                for ds, de in buffer_boundaries:
                    new_s = ds - max_length
                    new_e = de - max_length
                    if new_e > 0:
                        new_boundaries.append((new_s, new_e))

                buffer_boundaries = new_boundaries

    out_f.close()

    print("===== Finished =====")
    print(f"Total tokens processed: {total_tokens_processed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw jsonl input")
    parser.add_argument("--output", required=True, help="output jsonl")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--tokenizer_path", default="./tokenizer")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    prepare_samples(args.input, args.output, tokenizer, args.max_length)


if __name__ == "__main__":
    main()
