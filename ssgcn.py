import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GCNConv

class SSGCN(nn.Module):
    """
    Span-based Syntax Graph Convolutional Network for Aspect-based Sentiment Analysis.
    Combines multi-view Span Attention with syntax-guided Graph Convolutional Networks (GCN)
    to perform aspect-based sentiment analysis.
    """
    def __init__(self, args):
        super(SSGCN, self).__init__()
        self.args = args
        self.threshold = getattr(args, 'threshold', 2)    # 多视角Span窗口半径
        self.alpha = getattr(args, 'alpha', 0.8)          # DSW距离衰减因子

        # BERT编码器
        self.bert = BertModel.from_pretrained(args.bert_model)
        hidden_size = self.bert.config.hidden_size

        # BiLSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1,
                            bidirectional=True, batch_first=True)

        # Span Attention
        self.span_linear = nn.Linear(hidden_size, hidden_size)

        # 语法图卷积层（两层）
        self.gcn1 = GCNConv(hidden_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

        # 门控融合
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, args.num_classes)
        )

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        edge_index_list = inputs['edge_index']
        edge_weight_list = inputs['edge_weight']
        aspect_pos_list = inputs['aspect_pos']

        batch_size, seq_len = input_ids.size()

        # 1. BERT + BiLSTM编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_out = outputs.last_hidden_state
        lstm_out, _ = self.lstm(bert_out)

        span_reps = []
        syn_reps = []

        for i in range(batch_size):
            h = lstm_out[i]  # (seq_len, hidden)
            aspect_pos = aspect_pos_list[i]  # Aspect token位置

            # === 多视角Span Attention ===
            multi_span_outs = []
            for radius in range(self.threshold + 1):
                left = max(0, aspect_pos[0] - radius)
                right = min(seq_len - 1, aspect_pos[-1] + radius)

                mask = torch.zeros(seq_len, device=h.device)
                mask[left:right + 1] = 1.0

                masked_h = h * mask.unsqueeze(-1)

                aspect_mask = torch.zeros(seq_len, device=h.device)
                aspect_mask[aspect_pos] = 1.0
                aspect_vector = (masked_h * aspect_mask.unsqueeze(-1)).sum(dim=0) / (aspect_mask.sum() + 1e-8)

                attn_scores = torch.matmul(masked_h, self.span_linear(aspect_vector))
                attn_weights = torch.softmax(attn_scores, dim=0)

                span_out = torch.matmul(attn_weights.unsqueeze(0), masked_h).squeeze(0)
                multi_span_outs.append(span_out)

            multi_span_outs = torch.stack(multi_span_outs)
            span_out = multi_span_outs.mean(dim=0)
            span_reps.append(span_out)

            # === 语法图卷积 ===
            edge_index = edge_index_list[i]
            edge_weight = edge_weight_list[i]

            x = h
            x = F.relu(self.gcn1(x, edge_index, edge_weight))
            x = self.gcn2(x, edge_index, edge_weight)

            syn_out = x[aspect_pos].mean(dim=0)
            syn_reps.append(syn_out)

        # batch堆叠
        span_reps = torch.stack(span_reps)
        syn_reps = torch.stack(syn_reps)

        # === 门控融合 ===
        fusion = torch.cat([span_reps, syn_reps], dim=-1)
        gate = torch.sigmoid(self.gate(fusion))
        fused = gate * syn_reps + (1 - gate) * span_reps

        # === 分类 ===
        logits = self.classifier(fused)
        return logits