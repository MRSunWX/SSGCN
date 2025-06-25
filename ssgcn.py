import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GCNConv


class SSGCN(nn.Module):
    def __init__(self, args):
        super(SSGCN, self).__init__()
        self.args = args
        # 多视角邻近跨度最大半径（默认2）
        self.threshold = getattr(args, 'threshold', 2)

        # BERT编码器
        self.bert = BertModel.from_pretrained(args.bert_model)
        hidden_size = self.bert.config.hidden_size

        # BiLSTM捕获顺序上下文信息，双向LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1,
                            bidirectional=True, batch_first=True)

        # Span Attention中的线性变换
        self.span_linear = nn.Linear(hidden_size, hidden_size)

        # 语法GCN两层
        self.gcn1 = GCNConv(hidden_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

        # 门控融合层，将span和GCN输出融合为最终表示
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

        # 分类器，输出类别概率（例如三分类情感）
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, args.num_classes)
        )

    def forward(self, inputs):
        """
        inputs: dict，包含：
            - input_ids: BERT输入token ids，shape (batch_size, seq_len)
            - attention_mask: BERT attention mask
            - token_type_ids: BERT token type ids
            - edge_index: list，每个batch元素的图邻接矩阵(edge_index)
            - edge_weight: list，每个batch元素的边权重
            - aspect_pos: list，每个batch元素的aspect的token索引列表
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        edge_index_list = inputs['edge_index']
        edge_weight_list = inputs['edge_weight']
        aspect_pos_list = inputs['aspect_pos']

        batch_size, seq_len = input_ids.size()

        # 1. BERT编码器提取词向量表示
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_out = outputs.last_hidden_state  # (batch_size, seq_len, hidden)

        # 2. BiLSTM捕获顺序上下文
        lstm_out, _ = self.lstm(bert_out)  # (batch_size, seq_len, hidden)

        span_reps = []
        syn_reps = []

        # 对batch中每条样本单独处理（方便操作不同长度和邻接矩阵）
        for i in range(batch_size):
            h = lstm_out[i]  # (seq_len, hidden)
            aspect_pos = aspect_pos_list[i]  # aspect在句子中的token索引列表，如[3,4]

            # 3. 多视角邻近跨度增强的Span Attention
            multi_span_outs = []
            for radius in range(self.threshold + 1):
                # 邻近窗口范围
                left = max(0, aspect_pos[0] - radius)
                right = min(seq_len - 1, aspect_pos[-1] + radius)

                # 构造mask，邻近窗口内为1，其余为0
                mask = torch.zeros(seq_len, device=h.device)
                mask[left:right+1] = 1.0

                # mask作用于token表示，其他token置0
                masked_h = h * mask.unsqueeze(-1)  # (seq_len, hidden)

                # Aspect向量：aspect token对应的隐藏状态均值（加mask防止越界）
                aspect_mask = torch.zeros(seq_len, device=h.device)
                aspect_mask[aspect_pos] = 1.0
                aspect_vector = (masked_h * aspect_mask.unsqueeze(-1)).sum(dim=0) / (aspect_mask.sum() + 1e-8)

                # 计算Span Attention分数（加权求和）
                attn_scores = torch.matmul(masked_h, self.span_linear(aspect_vector))  # (seq_len,)
                attn_weights = torch.softmax(attn_scores, dim=0)
                span_out = torch.matmul(attn_weights.unsqueeze(0), masked_h).squeeze(0)  # (hidden,)
                multi_span_outs.append(span_out)

            # 多视角表示拼接后平均，融合不同邻近窗口信息
            multi_span_outs = torch.stack(multi_span_outs)  # (threshold+1, hidden)
            span_out = multi_span_outs.mean(dim=0)  # (hidden,)
            span_reps.append(span_out)

            # 4. Syntax-GCN，结合句法邻接矩阵和边权重进行图卷积
            edge_index = edge_index_list[i]
            edge_weight = edge_weight_list[i]

            x = h  # 输入节点特征
            x = F.relu(self.gcn1(x, edge_index, edge_weight))
            x = self.gcn2(x, edge_index, edge_weight)

            # Aspect节点表示均值作为GCN输出
            syn_out = x[aspect_pos].mean(dim=0)
            syn_reps.append(syn_out)

        # batch堆叠
        span_reps = torch.stack(span_reps)  # (batch, hidden)
        syn_reps = torch.stack(syn_reps)    # (batch, hidden)

        # 5. 门控融合span和GCN两路信息
        fusion = torch.cat([span_reps, syn_reps], dim=-1)  # (batch, hidden*2)
        gate = torch.sigmoid(self.gate(fusion))           # (batch, hidden)
        fused = gate * syn_reps + (1 - gate) * span_reps  # (batch, hidden)

        # 6. 分类器输出最终logits
        logits = self.classifier(fused)  # (batch, num_classes)
        return logits
