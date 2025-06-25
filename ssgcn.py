import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GCNConv


class SSGCN(nn.Module):
    def __init__(self, args):
        super(SSGCN, self).__init__()
        self.args = args

        # BERT Encoder
        self.bert = BertModel.from_pretrained(args.bert_model)
        hidden_size = self.bert.config.hidden_size

        # BiLSTM for capturing sequential context
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1,
                             bidirectional=True, batch_first=True)

        # Span Attention
        self.span_linear = nn.Linear(hidden_size, hidden_size)

        # Syntax-GCN (2 layers)
        self.gcn1 = GCNConv(hidden_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

        # Gating Mechanism for fusion
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

        # Classifier
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

        # BERT + BiLSTM Encoder
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
            h = lstm_out[i]

            # Span Attention
            aspect_pos = aspect_pos_list[i]
            aspect_vector = h[aspect_pos].mean(dim=0)

            attn_scores = torch.matmul(h, self.span_linear(aspect_vector))
            attn_weights = torch.softmax(attn_scores, dim=0)
            span_out = torch.matmul(attn_weights, h)
            span_reps.append(span_out)

            # Syntax-GCN
            edge_index = edge_index_list[i]
            edge_weight = edge_weight_list[i]

            x = h
            x = F.relu(self.gcn1(x, edge_index, edge_weight))
            x = self.gcn2(x, edge_index, edge_weight)

            syn_out = x[aspect_pos].mean(dim=0)
            syn_reps.append(syn_out)

        span_reps = torch.stack(span_reps)
        syn_reps = torch.stack(syn_reps)

        # Gated Fusion
        fusion = torch.cat([span_reps, syn_reps], dim=-1)
        gate = torch.sigmoid(self.gate(fusion))
        fused = gate * syn_reps + (1 - gate) * span_reps

        logits = self.classifier(fused)
        return logits
