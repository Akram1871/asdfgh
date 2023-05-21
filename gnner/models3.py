from turtle import forward
import torch
import torch.nn.functional as F
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch import nn
from transformers import AutoModel
from torch_geometric.utils import dense_to_sparse
from base import BaseModel
from torch_geometric.nn import GCNConv, GatedGraphConv, global_max_pool


class SpanInteractionAttention(nn.Module):

    def __init__(self, d_model, d_ffn=1024):

        super().__init__()

        self.d_model = d_model

        self.kq = FeedForward(d_model, num_layers=2, hidden_dims=[
                              d_ffn, d_model * 3], activations=[nn.ReLU(), nn.Identity()])

        self.scale = d_model ** -0.5

    def self_attention(self, x):

        k, q, v = self.kq(x).chunk(3, dim=-1)

        attention_weight = q @ k.transpose(1, 2) * self.scale

        return attention_weight, v

    def forward(self, x, interaction_mask=None):

        attention_weight, v = self.self_attention(x)

        if interaction_mask is not None:
            return attention_weight * interaction_mask @ v

        return attention_weight @ v


class GnnerAT(BaseModel):
    def __init__(self, labels, model_name, max_span_width=8, width_embedding_dim=128, project_dim=256):
        super().__init__(labels, model_name, max_span_width)

        self.num_labels = len(labels) + 1

        self.max_span_width = max_span_width

        self.encoder = AutoModel.from_pretrained(model_name)

        d_model = self.encoder.config.hidden_size

        self.span_ext = EndpointSpanExtractor(
            d_model, combination='x,y', num_width_embeddings=max_span_width, span_width_embedding_dim=width_embedding_dim, bucket_widths=True
        )

        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model * 2 + width_embedding_dim, project_dim),
        )

        self.graph_attention = SpanInteractionAttention(project_dim)

        self.fc = nn.Linear(256, self.num_labels)

    def forward(self, x):

        # compute hidden_state
        h = self.encoder(x['input_ids'], x['attention_mask']).last_hidden_state

        # span embeddings (concat endpoints)
        span_reps = self.span_ext.forward(
            h, x['span_ids'], span_indices_mask=x['span_mask'])

        # downproject span representation
        span_reps = self.project(span_reps)

        # gnn layer
        span_reps_with_graph = self.graph_attention.forward(
            span_reps, x['graph'])

        # GNN message passing
        logits = self.fc(span_reps_with_graph)

        output = {'logits': logits}

        # compute loss if training
        if self.training:
            labels = x['span_labels'].view(-1)
            output['loss'] = F.cross_entropy(
                logits.view(-1, self.num_labels), labels, ignore_index=-1, reduction='sum')

        return output


class GnnerCONV(BaseModel):
    def __init__(self, labels, model_name, max_span_width, width_embedding_dim, project_dim):
        super().__init__(labels, model_name, max_span_width)

        self.num_labels = len(labels) + 1

        self.max_span_width = max_span_width

        self.encoder = AutoModel.from_pretrained(model_name)

        d_model = self.encoder.config.hidden_size

        self.span_ext = EndpointSpanExtractor(
            d_model, combination='x,y', num_width_embeddings=max_span_width, span_width_embedding_dim=width_embedding_dim, bucket_widths=True
        )

        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model * 2 + width_embedding_dim, project_dim),
        )

        self.graph_attention = SpanInteractionAttention(project_dim)
        self.hidden_dim = int(project_dim/2)
        self.gcn_neg = GCNConv(project_dim,self.hidden_dim)
        self.gcn_neg2 = GCNConv(self.hidden_dim, project_dim)
        self.gcn_pos = nn.Linear(project_dim, project_dim)
        #self.gcn_pos = nn.LSTM(project_dim, project_dim // 2, bidirectional=True, batch_first=True)
        # Define max pooling layer
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=project_dim * 3)

        self.lstm = nn.LSTM(project_dim, project_dim // 2, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(project_dim * 3, self.num_labels)

    def forward(self, x):

        # compute hidden_state
        h = self.encoder(x['input_ids'], x['attention_mask']).last_hidden_state

        # span embeddings (concat endpoints)
        span_reps = self.span_ext.forward(
            h, x['span_ids'], span_indices_mask=x['span_mask'])

        # downproject span representation
        span_reps = self.project(span_reps)
        
        # GAT layer
        gat_output = self.graph_attention.forward(
            span_reps, x['graph'])

        graph = x['graph']
        B, NS, D = span_reps.size()
        edge_index, pos, neg = get_pos_neg(graph)

        # negative spans
        span_negs = self.gcn_neg(span_reps.view(-1, D), neg)

        span_negs = self.gcn_neg2(span_negs,neg)
        #span_negs = self.gcn_neg(span_reps, neg)
      
        #print(f"span_negs shape \t {span_negs.size()}")
        # max_pool = global_max_pool(span_negs, B)
        # print(f"max_pool shape \t {max_pool.size()}")
        # max_pool = max_pool.unsqueeze(0)
        # print(f"max_pool shape \t {max_pool.size()}")
        # h0 = torch.randn(4,self.max_len,self.hidden).to(self.device)
        # c0 = torch.randn(4,self.max_len,self.hidden).to(self.device)
        
        lstm, _ = self.lstm(span_negs)
        #print(f"Lstm shape \t {lstm.size()}")
        #maxpool_output = self.maxpool(lstm.permute(0, 2, 1)).squeeze()
        #print(f"max_pool shape \t {maxpool_output.size()}")
        # x = x.squeeze(0)
        # span_negs = span_negs.view(B, NS, -1)
        lstm = lstm.view(B, NS, -1)
        #print(f"Lstm shape \t {lstm.size()}")

        # positive spans
        span_pos = self.gcn_pos(span_reps)

        # concat pos and neg representation
        span_reps_with_graph = torch.cat([span_pos, lstm,gat_output], dim=-1)
        
        # Pass output through max pooling layer
        pooled_output = self.max_pool(span_reps_with_graph)

        # final layer
        logits = self.fc(pooled_output)

        output = {'logits': logits}
        output['batch'] = len(logits)

        # compute loss if training
        if self.training:
            labels = x['span_labels'].view(-1)
            output['loss'] = F.cross_entropy(
                logits.view(-1, self.num_labels), labels, ignore_index=-1, reduction='sum')

        return output


def get_pos_neg(graph):
    edge_idx, edge_feat = dense_to_sparse(graph)
    pos = edge_idx[:, edge_feat == 1]
    neg = edge_idx[:, edge_feat == -1]
    return edge_idx, pos, neg


class Baseline(BaseModel):
    def __init__(self, labels, model_name, max_span_width=8, width_embedding_dim=128, project_dim=512):
        super().__init__(labels, model_name, max_span_width)

        self.num_labels = len(labels) + 1

        self.max_span_width = max_span_width

        self.encoder = AutoModel.from_pretrained(model_name)

        d_model = self.encoder.config.hidden_size

        self.span_ext = EndpointSpanExtractor(
            d_model, combination='x,y', num_width_embeddings=max_span_width, span_width_embedding_dim=width_embedding_dim, bucket_widths=True
        )

        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model * 2 + width_embedding_dim, project_dim),
        )

        self.fc = nn.Linear(project_dim, self.num_labels)

    def forward(self, x):

        # compute hidden_state
        h = self.encoder(x['input_ids'], x['attention_mask']).last_hidden_state

        # span embeddings (concat endpoints)
        span_reps = self.span_ext.forward(
            h, x['span_ids'], span_indices_mask=x['span_mask'])

        # downproject span representation
        span_reps = self.project(span_reps)

        graph = x['graph']
        B, NS, D = span_reps.size()
        edge_index, pos, neg = get_pos_neg(graph)

        # negative spans
        span_negs = self.gcn_neg(span_reps.view(-1, D), neg)
        span_negs = span_negs.view(B, NS, -1)

        # positive spans
        span_pos = self.gcn_pos(span_reps)

        # concat pos and neg representation
        span_reps_with_graph = torch.cat([span_pos, span_negs], dim=-1)

        # final layer
        logits = self.fc(span_reps_with_graph)

        output = {'logits': logits}
        
        

        # compute loss if training
        if self.training:
            labels = x['span_labels'].view(-1)
            output['loss'] = F.cross_entropy(
                logits.view(-1, self.num_labels), labels, ignore_index=-1, reduction='sum')

        return output
