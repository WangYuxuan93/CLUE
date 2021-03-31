# coding=utf-8

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import PretrainedConfig, BertPreTrainedModel
from transformers.modeling_bert import (BertAttention, BertIntermediate, BertLayer, BertPooler)
from transformers.activations import ACT2FN

from models.gate import HighwayGateLayer

import logging
logger = logging.getLogger(__name__)


class SemSynBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        num_labels=2,
        fusion_type="joint",
        graph=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.num_labels = num_labels

        # GNN options
        self.fusion_type = fusion_type
        self.graph = graph


class PalBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_pal_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_pal_attention_heads)
            )

        input_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size

        self.num_pal_attention_heads = config.num_pal_attention_heads
        self.attention_head_size = int(input_size / config.num_pal_attention_heads)
        self.all_head_size = self.num_pal_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_pal_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class IntermediateGNNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GCNLayer(config)

        self.do_pal_project = config.graph["do_pal_project"]
        if config.graph["do_pal_project"]:
            self.dense_down = nn.Linear(config.hidden_size, config.graph["lowrank_size"])
            self.dense_up = nn.Linear(config.graph["lowrank_size"], config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.hidden_act_fn = ACT2FN[config.hidden_act]
        else:
            self.hidden_act_fn = config.hidden_act

    def forward(
        self, 
        hidden_states,
        attention_mask=None,
        heads=None,
        rels=None,
        debug=False,
    ):
        if self.do_pal_project:
            hidden_states = self.dense_down(hidden_states)
        
        hidden_states = self.attention(hidden_states, attention_mask, heads=heads, rels=rels)

        if self.do_pal_project:
            hidden_states = self.dense_up(hidden_states)

        hidden_states = self.hidden_act_fn(hidden_states)

        return hidden_states


class GCNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_self_weight = config.graph["use_self_weight"]
        self.use_rel_embedding = config.graph["use_rel_embedding"]

        input_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size
        output_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size
        self.data_flow = config.graph["data_flow"]
        self.adj_weight = nn.Linear(input_size, output_size)
        if self.use_self_weight:
            self.self_weight = nn.Linear(input_size, output_size)
        if self.data_flow == "bidir":
            self.reverse_adj_weight = nn.Linear(input_size, output_size)

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
        if self.data_flow == "c2h":
            # remask pads at the end of each row because it's permuted
            adj_matrix = adj_matrix.permute(0,2,1)

        if debug:
            torch.set_printoptions(profile="full")
            print ("adj_matrix:\n", adj_matrix)

        # (batch, seq_len, output_size)
        context_layer = torch.matmul(adj_matrix, adj_layer)
        if self.data_flow == "bidir":
            reverse_adj_layer = self.reverse_adj_weight(hidden_states)
            reverse_adj_matrix = adj_matrix.permute(0,2,1)
            reverse_context_layer = torch.matmul(reverse_adj_matrix, reverse_adj_layer)


        # divide by the number of neighbors
        # (batch, seq_len)
        num_neighbors = adj_matrix.sum(-1)
        ones = torch.ones_like(num_neighbors, device=context_layer.device)
        num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
        if debug:
            print ("num_neighbors:\n", num_neighbors)
        # divide by the number of neighbors
        context_layer = context_layer / num_neighbors.unsqueeze(-1)
        if self.data_flow == "bidir":
            num_neighbors = reverse_adj_matrix.sum(-1)
            ones = torch.ones_like(num_neighbors, device=context_layer.device)
            num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
            # divide by the number of neighbors
            reverse_context_layer = reverse_context_layer / num_neighbors.unsqueeze(-1)
            context_layer = context_layer + reverse_context_layer

        if self.use_self_weight:
            self_layer = self.self_weight(hidden_states)
            context_layer = self_layer + context_layer

        # context_layer = [self_layer] + adj_layer + [rev_adj_layer]
        return context_layer


class ResidualGNNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.do_pal_project = config.graph["do_pal_project"]
        self.encoder_type = config.graph["encoder"]
        
        if isinstance(config.hidden_act, str):
            self.hidden_act_fn = ACT2FN[config.hidden_act]
        else:
            self.hidden_act_fn = config.hidden_act

        if config.graph["do_pal_project"]:
            self.dense_down = nn.Linear(config.hidden_size, config.graph["lowrank_size"])
            #output_size = 2*config.graph["lowrank_size"] if (config.graph["use_rel_embedding"] and config.rel_combine_type=="concat") else config.graph["lowrank_size"]
            output_size = config.graph["lowrank_size"]
            self.dense_up = nn.Linear(output_size, config.hidden_size)
        
        if self.encoder_type == "GCN":
            self.attention = GCNLayer(config)
        elif self.encoder_type == "ATT": # vanilla attention
            self.attention = PalBertSelfAttention(config)
        elif self.encoder_type == "LIN": # linear 
            self.attention = None


    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        heads=None, 
        rels=None,
    ):
        if self.do_pal_project:
            hidden_states = self.dense_down(hidden_states)
        if self.encoder_type == "GCN":
            hidden_states = self.attention(hidden_states, attention_mask, heads=heads, rels=rels)
        elif self.encoder_type == "ATT":
            hidden_states = self.attention(hidden_states, attention_mask)
        elif self.encoder_type == "LIN":
            # for linear we add act in between
            hidden_states = self.hidden_act_fn(hidden_states)
        
        if self.do_pal_project:
            hidden_states = self.dense_up(hidden_states)
        if self.encoder_type != "LIN":
            hidden_states = self.hidden_act_fn(hidden_states)
        
        return hidden_states


class ResidualBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_fusion_gate = config.graph["use_fusion_gate"]
        if self.use_fusion_gate:
            self.gate = eval(config.graph["residual_fusion_gate"])(config.hidden_size)

    def forward(self, hidden_states, input_tensor, res_layer):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.use_fusion_gate:
            hidden_states = self.gate(hidden_states + input_tensor, res_layer)
        else:
            hidden_states = hidden_states + input_tensor + res_layer
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ResidualGNNBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = ResidualBertOutput(config)

        self.fusion_type = config.fusion_type
        self.res_layer = ResidualGNNLayer(config)
        """
        if self.pal_gate_type == "scalar":
            # no need to init
            self.pal_gate_scalar = nn.Parameter(torch.tensor(0.))
        elif self.pal_gate_type == "task":
            #self.pal_gamma = nn.Parameter(torch.FloatTensor([10.0]))
            self.pal_gate_dense = nn.Linear(config.task_embed_size, 1)
        elif self.pal_gate_type == "input":
            self.pal_gate_dense = nn.Linear(config.hidden_size, 1, bias=False)
        """
    """
    def task_gate_value(self, task_embed=None):
        if self.pal_gate_type == "scalar":   
            return nn.Sigmoid()(self.pal_gate_scalar)
        elif self.pal_gate_type == "task":
            #return self.pal_gate_dense(task_embed)
            return nn.Sigmoid()(self.pal_gate_dense(task_embed))
        else:
            return None
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=True,
        heads=None,
        rels=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        res_output = self.res_layer(hidden_states, attention_mask, heads, rels)
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, res_output)

        outputs = (layer_output,) + outputs
        return outputs


class SemSynBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion_type = config.graph["fusion_type"]
        self.structured_layers = config.graph["structured_layers"] if config.graph["structured_layers"] is not None else [i for i in range(config.num_hidden_layers)]

        if self.fusion_type in ["inter", "top"]:
            self.layer = nn.ModuleList([BertLayer(config) for i in range(config.num_hidden_layers)])
        elif self.fusion_type == "residual":
            self.layer = nn.ModuleList([ResidualGNNBertLayer(config) if i in self.structured_layers
                                    else BertLayer(config) for i in range(config.num_hidden_layers)])

        self.inter_gnn_layers = None
        if self.fusion_type == "inter":
            self.inter_gnn_layers = nn.ModuleList([IntermediateGNNLayer(config) if i in self.structured_layers
                                    else None for i in range(config.num_hidden_layers)])
            self.use_fusion_gate = config.graph["use_fusion_gate"]
            if self.use_fusion_gate:
                self.gate = eval(config.graph["inter_fusion_gate"])(config.hidden_size)

    """
    def task_gate_values(self):
        if not ((self.aug_type == "pal" and self.pal_gate_type in ["scalar","share-scalar","task"]
            ) or (self.aug_type == "inter_gnn" and self.inter_gnn_gate_type in ["scalar","share-scalar"])):
            return None
        vals = []
        if self.aug_type == "inter_gnn" and self.inter_gnn_gate_type == "share-scalar":
            return float(nn.Sigmoid()(self.inter_gnn_pal_gate_scalar).detach().cpu().numpy())
        if self.aug_type == "pal" and self.pal_gate_type == "share-scalar":
            return float(nn.Sigmoid()(self.pal_gate_scalar).detach().cpu().numpy())
            #return float(self.pal_gate_scalar.detach().cpu().numpy())
        for i, layer in enumerate(self.layer):
            if i in self.structured_layers:
                if self.aug_type == "pal":
                    vals.append(float(layer.res_layer.task_gate_value(self.task_embed).detach().cpu().numpy()))
                elif self.aug_type == "inter_gnn":
                    vals.append(float(self.inter_gnn_layers[i].task_gate_value().detach().cpu().numpy()))

        return vals
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=False,
        heads=None,
        rels=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                if self.fusion_type != "inter" and i in self.structured_layers:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        output_attentions,
                        heads,
                        rels
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        output_attentions,
                    )
            hidden_states = layer_outputs[0]

            if self.fusion_type == "inter" and i in self.structured_layers:
                inter_hidden_states = self.inter_gnn_layers[i](
                    hidden_states,
                    attention_mask,
                    heads,
                    rels,
                )
                if self.use_fusion_gate:
                    hidden_states = self.gate(hidden_states, inter_hidden_states)
                else:
                    hidden_states = 0.5*(hidden_states + inter_hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class SemSynBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = SemSynBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]


class SemSynBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_other_parser = config.use_other_parser

        self.bert = SemSynBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        controller="main",
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if self.use_other_parser:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class SemSynBertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = SemSynBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels

        heads = heads.view(-1, heads.size(-2), heads.size(-1)) if heads is not None else None
        rels = rels.view(-1, rels.size(-2), rels.size(-1)) if rels is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


class SemSynBertForQuestionAnswering(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = SemSynBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output