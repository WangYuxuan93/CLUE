import torch
from torch import nn

class BiaffineAttention(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()
    
    def reset_parameters(self):
        #nn.init.zeros_(self.weight)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, y):
        """
        matmul(x, w)
        x: [batch_size, 1, seq_len, h] => [batch_size, n_out, seq_len, h], stack n_out times
        w: [n_out, h, h]) => [batch_size, n_out, h, h], stack batch_size times
        output: [batch_size, n_out, seq_len, h]
        """
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1]).unsqueeze(-1)], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1]).unsqueeze(-1)], -1)
        # [batch_size, 1, seq_len, n_in+bias_x]
        x = x.unsqueeze(1)
        # [batch_size, 1, seq_len, n_in+bias_y]
        y = y.unsqueeze(1)
        # [batch_size, n_out, seq_len, seq_len]
        # s = torch.matmul(torch.matmul(x, self.weight), y.transpose(-1, -2))
        s = x @ self.weight @ y.transpose(-1, -2)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)
        return s


class BiaffineArcAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.arc_attention_head_size = config.arc_attention_head_size
        self.arc_attention_mode = config.arc_attention_mode
        self.query = nn.Linear(config.hidden_size, config.arc_mlp_size)
        self.key = nn.Linear(config.hidden_size, config.arc_mlp_size)
        self.value = nn.Linear(config.hidden_size, self.arc_attention_head_size)

        self.dropout = nn.Dropout(config.arc_attention_probs_dropout_prob)

        self.biaffine = BiaffineAttention(config.arc_mlp_size, bias_x=True, bias_y=False)

    def forward(
        self, 
        hidden_states,
        attention_mask=None,
        output_attentions=True,
    ):
        # arc-dep h
        query_layer = self.query(hidden_states)
        # arc-head h
        key_layer = self.key(hidden_states)
        # (batch, seq_len, arc_att_head_size)
        value_layer = self.value(hidden_states)

        # (batch, seq_len, seq_len)
        attention_scores = self.biaffine(query_layer, key_layer)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # attention_mask (batch, 1, 1, seq_len) => (batch, 1, seq_len)
            # (batch, seq_len, seq_len)
            attention_scores = attention_scores + attention_mask.squeeze(1)
        #print ("attention_scores:\n", attention_scores)
        # dependent to modifier
        #if self.arc_attention_mode == "dep2mod":
        # Normalize the attention scores to probabilities.
        # (batch, seq_len, seq_len)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # modifier to dependent, this cause multi-heads
        if self.arc_attention_mode == "c2h":
            # remask pads at the end of each row because it's permuted
            attention_probs = nn.Softmax(dim=-1)(attention_probs.permute(0,2,1)+attention_mask.squeeze(1))

        seq_len = list(attention_probs.size())[1]
        # diagnose: (seq_len, seq_len) => (1, seq_len, seq_len)
        diag_tensor = torch.diag(torch.ones(seq_len, device=hidden_states.device)).unsqueeze(0)
        # set entries on diagonal to 1
        attention_probs = torch.where(diag_tensor == 1, diag_tensor, attention_probs)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # (batch, seq_len, arc_att_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)

        # return attention_scores for parsing loss
        outputs = (context_layer, attention_scores.permute(0,2,1)) if output_attentions else (context_layer,)
        return outputs


class BiaffineRelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rel_attention_head_size = config.rel_attention_head_size

        self.query = nn.Linear(config.hidden_size, config.rel_mlp_size)
        self.key = nn.Linear(config.hidden_size, config.rel_mlp_size)
        #self.value = nn.Linear(config.hidden_size, self.rel_attention_head_size)

        self.dropout = nn.Dropout(config.rel_attention_probs_dropout_prob)
        self.biaffine = BiaffineAttention(config.rel_mlp_size, n_out=config.num_rel_labels, bias_x=True, bias_y=True)

        self.rel_embed_matrix = nn.Parameter(torch.Tensor(config.num_rel_labels, config.rel_attention_head_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        #nn.init.zeros_(self.weight)
        nn.init.xavier_uniform_(self.rel_embed_matrix)

    def forward(
        self, 
        hidden_states,
        attention_mask=None,
        output_attentions=True,
    ):
        # rel-dep h
        query_layer = self.query(hidden_states)
        # rel-head h
        key_layer = self.key(hidden_states)

        # (batch, n_rels, seq_len, seq_len)
        attention_scores = self.biaffine(query_layer, key_layer)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # (batch, seq_len, seq_len, n_rels)
        attention_probs = nn.Softmax(dim=-1)(attention_scores.permute(0,2,3,1))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # (batch, seq_len, seq_len, n_rels) x (n_rels, rel_att_head_size)
        # => (batch, seq_len, seq_len, rel_att_head_size)
        context_layer = torch.matmul(attention_probs, self.rel_embed_matrix)

        # (batch, seq_len, rel_att_head_size)
        context_layer = context_layer.mean(-2).squeeze(-2)

        # return attention_scores for parsing loss
        outputs = (context_layer, attention_scores) if output_attentions else (context_layer,)
        return outputs