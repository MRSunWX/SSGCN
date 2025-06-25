import os
import json
import numpy as np
import pickle
import spacy
from collections import deque
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset

# ================== Tokenizer 类 ==================
class Tokenizer:
    def __init__(self, pretrained_model='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length

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
        aspect_tokens = aspect.split()
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i + len(aspect_tokens)] == aspect_tokens:
                return list(range(i, i + len(aspect_tokens)))
        print(f"[Warning] Aspect未对齐: {aspect} in {tokens}")
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
