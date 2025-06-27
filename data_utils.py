import os
import json
import numpy as np
import pickle
import spacy
from collections import deque
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import glob
import argparse
from fuzzywuzzy import fuzz  # 引入 fuzzywuzzy 进行模糊匹配

# ================== Tokenizer 类 ==================
class Tokenizer:
    def __init__(self, pretrained_model='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length
        self.nlp = spacy.load('en_core_web_sm')  # 用于词形还原

    def encode(self, sentence, aspect):
        encoded = self.tokenizer.encode_plus(
            sentence,
            aspect,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0)
        }

    def get_aspect_position(self, tokens: list, aspect: str):
        """
        改进的 aspect 定位方法，解决未对齐问题。
        返回 aspect 在 tokens 中的位置（索引列表）。
        """
        # 将 tokens 和 aspect 转换为小写以进行大小写无关匹配
        aspect_tokens = aspect.lower().split()
        tokens_lower = [t.lower() for t in tokens]

        # 方法 1：精确匹配
        for i in range(len(tokens_lower) - len(aspect_tokens) + 1):
            if tokens_lower[i:i + len(aspect_tokens)] == aspect_tokens:
                return list(range(i, i + len(aspect_tokens)))

        # 方法 2：模糊匹配（使用 fuzzywuzzy 比较相似度）
        best_score = 0
        best_pos = [0]
        for i in range(len(tokens_lower) - len(aspect_tokens) + 1):
            candidate = ' '.join(tokens_lower[i:i + len(aspect_tokens)])
            score = fuzz.ratio(candidate, aspect.lower())
            if score > best_score and score > 80:  # 阈值 80 可调整
                best_score = score
                best_pos = list(range(i, i + len(aspect_tokens)))

        if best_score > 80:
            print(f"[Info] 模糊匹配成功: aspect='{aspect}', tokens={tokens[i:i + len(aspect_tokens)]}, score={best_score}")
            return best_pos

        # 方法 3：词形还原匹配
        doc = self.nlp(' '.join(tokens))
        tokens_lemmatized = [tok.lemma_.lower() for tok in doc]
        aspect_doc = self.nlp(aspect.lower())
        aspect_lemmatized = [tok.lemma_.lower() for tok in aspect_doc]

        for i in range(len(tokens_lemmatized) - len(aspect_lemmatized) + 1):
            if tokens_lemmatized[i:i + len(aspect_lemmatized)] == aspect_lemmatized:
                return list(range(i, i + len(aspect_lemmatized)))

        # 方法 4：单词部分匹配（当 aspect 是短语的子集）
        for i in range(len(tokens_lower)):
            if any(at in tokens_lower[i] for at in aspect_tokens):
                print(f"[Info] 部分匹配: aspect='{aspect}', matched token='{tokens[i]}' at index {i}")
                return [i]

        # 最终回退：返回 [0] 并记录警告
        print(f"[Warning] Aspect未对齐: aspect='{aspect}', tokens={tokens}")
        return [0]

# ================== Dependency Graph 类 ==================
class DependencyGraph:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def parse(self, sentence: str):
        doc = self.nlp(sentence)
        tokens = [tok.text for tok in doc]
        edges = []
        for tok in doc:
            if tok.i == tok.head.i:
                continue
            edges.append((tok.head.i, tok.i))
            edges.append((tok.i, tok.head.i))
        return tokens, edges

    def build_aodt(self, edges, aspect_set, max_distance=3):
        graph = {}
        for u, v in edges:
            graph.setdefault(u, []).append(v)

        visited = set(aspect_set)
        queue = deque([(idx, 0) for idx in aspect_set])

        while queue:
            node, dist = queue.popleft()
            if dist >= max_distance:
                continue
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return visited

    def shortest_distance(self, edges, node_num):
        inf = 1e9
        dist = np.full((node_num, node_num), inf)
        np.fill_diagonal(dist, 0)
        for u, v in edges:
            dist[u, v] = 1
        for k in range(node_num):
            for i in range(node_num):
                for j in range(node_num):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    def compute_dsw(self, dist_matrix, alpha=1.0):
        weights = np.exp(-alpha * dist_matrix)
        weights[dist_matrix >= 1e9] = 0
        return weights

    def build_graph(self, edges, node_num, aspect_set, alpha=1.0, max_distance=3):
        nodes = self.build_aodt(edges, aspect_set, max_distance)
        sub_edges = [(u, v) for u, v in edges if u in nodes and v in nodes]

        dist = self.shortest_distance(sub_edges, node_num)
        weight = self.compute_dsw(dist, alpha)

        edge_index = np.array(np.nonzero(weight))
        edge_weight = weight[edge_index[0], edge_index[1]]
        return edge_index, edge_weight

# ================== Dataset 和 DataLoader ==================
class ABSADataset(Dataset):
    def __init__(self, data, tokenizer, dep_parser, max_seq_len=128, alpha=1.0, max_dep_dist=3):
        self.tokenizer = tokenizer
        self.dep_parser = dep_parser
        self.max_seq_len = max_seq_len
        self.alpha = alpha
        self.max_dep_dist = max_dep_dist

        self.data = []
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        for d in data:
            tokens = d['token']
            aspect = ' '.join(d['aspect'])
            polarity = d['polarity']
            label = label_map[polarity]

            sentence = ' '.join(tokens)
            tokens_parsed, edges = self.dep_parser.parse(sentence)
            aspect_pos = self.tokenizer.get_aspect_position(tokens_parsed, aspect)
            edge_index, edge_weight = self.dep_parser.build_graph(
                edges, node_num=len(tokens_parsed), aspect_set=set(aspect_pos),
                alpha=self.alpha, max_distance=self.max_dep_dist
            )

            bert_input = self.tokenizer.encode(sentence, aspect)

            self.data.append({
                'input_ids': bert_input['input_ids'],
                'attention_mask': bert_input['attention_mask'],
                'token_type_ids': bert_input['token_type_ids'],
                'aspect_pos': torch.tensor(aspect_pos, dtype=torch.long),
                'edge_index': torch.tensor(edge_index, dtype=torch.long),
                'edge_weight': torch.tensor(edge_weight, dtype=torch.float),
                'label': torch.tensor(label, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch_out = {}
    keys = batch[0].keys()
    for key in keys:
        if key in ['edge_index', 'edge_weight', 'aspect_pos']:
            batch_out[key] = [item[key] for item in batch]
        else:
            batch_out[key] = torch.stack([item[key] for item in batch])
    return batch_out

def load_json_data(json_file):
    """
    Load JSON file and convert to ABSADataset-compatible format.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        processed_data = []
        for item in json_data:
            sentence = item['sentence']
            tokens = sentence.split()
            aspect = item['aspect'].split()
            polarity = item['label']
            
            processed_data.append({
                'token': tokens,
                'aspect': aspect,
                'polarity': polarity
            })
        
        return processed_data
    except Exception as e:
        print(f"Error loading {json_file}: {str(e)}")
        return []

def save_dataset(dataset, output_file):
    """
    Save the processed dataset to a pickle file.
    """
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Successfully saved dataset to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {str(e)}")

def process_json_files(input_dir, output_dir, max_seq_len=128, alpha=1.0, max_dep_dist=3):
    """
    Process all JSON files in the input directory and save as pickle files.
    """
    tokenizer = Tokenizer(pretrained_model='bert-base-uncased', max_length=max_seq_len)
    dep_parser = DependencyGraph()

    if not os.path.isdir(input_dir):
        print(f"Error: Directory {input_dir} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}.")
        return

    for json_file in json_files:
        print(f"Processing {json_file}...")
        data = load_json_data(json_file)
        if not data:
            print(f"No valid data in {json_file}. Skipping.")
            continue

        try:
            dataset = ABSADataset(
                data,
                tokenizer=tokenizer,
                dep_parser=dep_parser,
                max_seq_len=max_seq_len,
                alpha=alpha,
                max_dep_dist=max_dep_dist
            )
        except Exception as e:
            print(f"Error creating dataset for {json_file}: {str(e)}")
            continue

        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + '.pkl')
        save_dataset(dataset, output_file)

def main():
    parser = argparse.ArgumentParser(description="Process SemEval JSON files into ABSADataset pickle files.")
    parser.add_argument('--input-dir', default='./dataset/semeval14',
                        help='Input directory containing JSON files')
    parser.add_argument('--max-seq-len', type=int, default=128,
                        help='Maximum sequence length for BERT tokenizer')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha parameter for dependency graph weights')
    parser.add_argument('--max-dep-dist', type=int, default=3,
                        help='Maximum dependency distance for graph construction')

    args = parser.parse_args()

    # 固定输出目录为 dataset/processed/
    output_dir = './dataset/processed'

    # Process JSON files
    process_json_files(
        input_dir=args.input_dir,
        output_dir=output_dir,
        max_seq_len=args.max_seq_len,
        alpha=args.alpha,
        max_dep_dist=args.max_dep_dist
    )

if __name__ == "__main__":
    main()