import os
import argparse
import pickle
import numpy as np
from collections import Counter

class VocabHelp:
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            counter.pop(tok, None)
        words_and_frequencies = sorted(counter.items(), key=lambda x: x[0])
        words_and_frequencies.sort(key=lambda x: x[1], reverse=True)
        for word, freq in words_and_frequencies:
            self.itos.append(word)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def read_dagcn_tokens(data_dir):
    files = ['train.txt', 'valid.txt', 'test.txt']
    all_tokens = []
    for file in files:
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                sentence = parts[0]
                tokens = sentence.strip().split()
                all_tokens.extend(tokens)
    return all_tokens

def build_vocab(tokens):
    return Counter(tokens)

def load_glove(glove_file, vocab: VocabHelp, dim=300):
    embed = np.random.uniform(-0.25, 0.25, (len(vocab), dim)).astype(np.float32)
    found = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.array(list(map(float, parts[1:])), dtype=np.float32)
            if word in vocab.stoi:
                embed[vocab.stoi[word]] = vec
                found += 1
    print(f"[GloVe] Matched {found}/{len(vocab)} words.")
    return embed

def main():
    parser = argparse.ArgumentParser(description="Build vocab and glove embedding from DAGCN-format dataset")
    parser.add_argument('--data_dir', required=True, help="Path to DAGCN-style dataset directory")
    parser.add_argument('--vocab_out', default='vocab.pkl')
    parser.add_argument('--glove_file', required=True, help="Path to glove.840B.300d.txt")
    parser.add_argument('--embed_out', default='glove_embed.npy')
    parser.add_argument('--emb_dim', type=int, default=300)
    args = parser.parse_args()

    print("[Step 1] Reading DAGCN-style tokens from dataset")
    tokens = read_dagcn_tokens(args.data_dir)
    counter = build_vocab(tokens)

    print("[Step 2] Building vocab")
    vocab = VocabHelp(counter)
    vocab.save(args.vocab_out)
    print(f"[Saved] Vocab size: {len(vocab)} → {args.vocab_out}")

    print("[Step 3] Loading GloVe and building embedding matrix")
    emb = load_glove(args.glove_file, vocab, args.emb_dim)
    np.save(args.embed_out, emb)
    print(f"[Saved] Embedding matrix → {args.embed_out}")

if __name__ == '__main__':
    main()
