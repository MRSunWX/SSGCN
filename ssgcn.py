import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GCNConv


class SSGCN(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', hidden_dim=300, gcn_hidden=300, num_classes=3, dropout=0.5):
        super(SSGCN, self).__init__()

        # BERT + BiLSTM Encoder
        self.bert = BertModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)

        # Span-Attention 分支
        self.span_linear = nn.Linear(hidden_dim, hidden_dim)

        # Syntax-GCN 分支
        self.gcn1 = GCNConv(hidden_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, hidden_dim)

        # 融合门控
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids,
                edge_index_list, edge_weight_list, aspect_pos_list):
        
        batch_size, seq_len = input_ids.size()

        # BERT + BiLSTM 编码
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_out = outputs.last_hidden_state  # [batch, seq_len, hidden]

        lstm_out, _ = self.lstm(bert_out)     # [batch, seq_len, hidden]

        span_reps = []
        syn_reps = []

        for i in range(batch_size):
            h = lstm_out[i]  # [seq_len, hidden]

            # -------- Span-Attention --------
            aspect_pos = aspect_pos_list[i]  # list of token positions
            aspect_vector = h[aspect_pos].mean(dim=0)  # [hidden]

            attn_scores = torch.matmul(h, self.span_linear(aspect_vector))  # [seq_len]
            attn_weights = F.softmax(attn_scores, dim=0)
            span_out = torch.matmul(attn_weights, h)  # [hidden]
            span_reps.append(span_out)

            # -------- Syntax-GCN --------
            edge_index = edge_index_list[i]  # [2, num_edges]
            edge_weight = edge_weight_list[i]  # [num_edges]

            x = h  # node features
            x = F.relu(self.gcn1(x, edge_index, edge_weight))
            x = self.gcn2(x, edge_index, edge_weight)

            syn_out = x[aspect_pos].mean(dim=0)  # 聚合aspect邻域
            syn_reps.append(syn_out)

        span_reps = torch.stack(span_reps)  # [batch, hidden]
        syn_reps = torch.stack(syn_reps)    # [batch, hidden]

        # -------- 融合 --------
        fusion = torch.cat([span_reps, syn_reps], dim=-1)
        gate = torch.sigmoid(self.gate(fusion))
        fused = gate * syn_reps + (1 - gate) * span_reps

        logits = self.classifier(fused)
        return logits


if __name__ == '__main__':
    model = SSGCN()
    input_ids = torch.randint(0, 100, (2, 50))
    attention_mask = torch.ones(2, 50)
    token_type_ids = torch.zeros(2, 50)

    edge_index_list = [torch.randint(0, 50, (2, 120)), torch.randint(0, 50, (2, 130))]
    edge_weight_list = [torch.rand(120), torch.rand(130)]
    aspect_pos_list = [torch.tensor([3, 4]), torch.tensor([7])]

    logits = model(input_ids, attention_mask, token_type_ids,
                   edge_index_list, edge_weight_list, aspect_pos_list)
    print(logits)
