# coding=utf-8

import math
import os
import warnings

import torch
import torch.utils.checkpoint
from torch import nn

from .configuration_bert import BertConfig

import logging
logger = logging.get_logger(__name__)


class GCNBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_self_weight = config.use_self_weight
        self.use_rel_embedding = config.use_rel_embedding
        self.use_dist = config.use_dist
        #self.arc_attention_head_size = config.arc_attention_head_size
        input_size = config.lowrank_size if config.do_pal_project else config.hidden_size
        output_size = config.lowrank_size if config.do_pal_project else config.hidden_size
        self.arc_attention_mode = config.arc_attention_mode
        self.adj_weight = nn.Linear(input_size, output_size)
        if self.use_self_weight:
            self.self_weight = nn.Linear(input_size, output_size)
        if self.arc_attention_mode == "bidir":
            self.reverse_adj_weight = nn.Linear(input_size, output_size)

        if self.use_rel_embedding:
            self.rel_combine_type = config.rel_combine_type
            self.use_reverse_rel_embed = config.use_reverse_rel_embed
            self.use_reverse_rel_weight = config.use_reverse_rel_weight
            if self.use_reverse_rel_embed:
                self.num_base_rels = config.num_rel_labels // 2
            self.rel_embeddings = nn.Embedding(config.num_rel_labels, config.rel_embed_size, padding_idx=0)
            self.rel_weight = nn.Linear(config.rel_embed_size, output_size)
            if self.use_reverse_rel_weight:
                self.reverse_rel_weight = nn.Linear(config.rel_embed_size, output_size)
            self.output = None
            # if not projected we have to make the output back to hidden_size
            if self.rel_combine_type=="concat" and not config.do_pal_project:
                self.output = nn.Linear(2*config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.arc_attention_probs_dropout_prob)

    def forward(
        self, 
        hidden_states,
        attention_mask=None,
        heads=None,
        rels=None,
        debug=False,
    ):
        
        # (batch, seq_len, arc_att_head_size)
        adj_layer = self.adj_weight(hidden_states)

        # (batch, seq_len, seq_len)
        # use the predicted heads from other parser
        adj_matrix = heads.float()
        # modifier to dependent, this cause multi-heads
        if self.arc_attention_mode == "c2h":
            # remask pads at the end of each row because it's permuted
            adj_matrix = adj_matrix.permute(0,2,1)
        # if use distance matrix, normalize them
        if self.use_dist:
            adj_matrix = adj_matrix / (1e-8+adj_matrix.sum(-1).unsqueeze(-1))

        if debug:
            torch.set_printoptions(profile="full")
            print ("adj_matrix:\n", adj_matrix)

        # (batch, seq_len, output_size)
        context_layer = torch.matmul(adj_matrix, adj_layer)
        if self.arc_attention_mode == "bidir":
            reverse_adj_layer = self.reverse_adj_weight(hidden_states)
            reverse_adj_matrix = adj_matrix.permute(0,2,1)
            if self.use_dist:
                reverse_adj_matrix = reverse_adj_matrix / (1e-8+reverse_adj_matrix.sum(-1).unsqueeze(-1))
            reverse_context_layer = torch.matmul(reverse_adj_matrix, reverse_adj_layer)

        # do not divide when using distance matrix
        if not self.use_dist:
            # divide by the number of neighbors
            # (batch, seq_len)
            num_neighbors = adj_matrix.sum(-1)
            ones = torch.ones_like(num_neighbors, device=context_layer.device)
            num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
            if debug:
                print ("num_neighbors:\n", num_neighbors)
            # divide by the number of neighbors
            context_layer = context_layer / num_neighbors.unsqueeze(-1)
        if self.arc_attention_mode == "bidir":
            # do not divide when using distance matrix
            if not self.use_dist:
                num_neighbors = reverse_adj_matrix.sum(-1)
                ones = torch.ones_like(num_neighbors, device=context_layer.device)
                num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
                # divide by the number of neighbors
                reverse_context_layer = reverse_context_layer / num_neighbors.unsqueeze(-1)
            context_layer = context_layer + reverse_context_layer

        if self.use_self_weight:
            self_layer = self.self_weight(hidden_states)
            context_layer = self_layer + context_layer

        if self.use_rel_embedding:
            # (batch, seq_len, seq_len, rel_embed_size)
            rel_embeds = self.rel_embeddings(rels)
            rel_embeds = self.dropout(rel_embeds)
            # (batch, seq_len, seq_len, output_size)
            # mask out rels with no arc
            rel_layer = self.rel_weight(rel_embeds) * adj_matrix.unsqueeze(-1)
            # (batch, seq_len, output_size)
            rel_layer = rel_layer.sum(-2)
            if self.use_reverse_rel_embed:
                zeros = torch.zeros_like(rels, dtype=torch.long, device=rels.device)
                reverse_rels = rels.permute(0,2,1) + self.num_base_rels
                reverse_rels = torch.where(reverse_rels>self.num_base_rels, reverse_rels, zeros)
                reverse_rel_embeds = self.rel_embeddings(reverse_rels)
                reverse_rel_embeds = self.dropout(reverse_rel_embeds)
                reverse_adj_matrix = adj_matrix.permute(0,2,1)
                #print ("reverse_rels:\n", reverse_rels)
                if self.use_reverse_rel_weight:
                    # mask out rels with no arc
                    reverse_rel_layer = self.reverse_rel_weight(reverse_rel_embeds) * reverse_adj_matrix.unsqueeze(-1)
                else:
                    reverse_rel_layer = self.rel_weight(reverse_rel_embeds) * reverse_adj_matrix.unsqueeze(-1)
                # (batch, seq_len, output_size)
                reverse_rel_layer = reverse_rel_layer.sum(-2)
                rel_layer = rel_layer + reverse_rel_layer

            
            if self.rel_combine_type=="concat":
                # (batch, seq_len, 2*output_size)
                context_layer = torch.cat([context_layer,rel_layer], dim=-1)
                if self.output is not None:
                    # (batch, seq_len, hidden_size)
                    context_layer = self.output(context_layer)
            else: #self.rel_combine_type=="add":
                context_layer = context_layer + rel_layer

        # context_layer = [self_layer] + adj_layer + [rev_adj_layer] + [rel_layer + [rev_rel_layer]]
        return context_layer
