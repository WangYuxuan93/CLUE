class RGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.lowrank_size if config.do_pal_project else config.hidden_size
        self.output_size = config.lowrank_size if config.do_pal_project else config.hidden_size
        self.num_basic_matrix = config.num_basic_matrix
        self.num_rel_labels = config.num_rel_labels
        self.basic_matrix = nn.Parameter(torch.Tensor(self.num_basic_matrix,
                                                self.input_size,
                                                self.output_size))
        self.rel_weight = nn.Parameter(torch.Tensor(self.num_rel_labels,
                                                self.num_basic_matrix))
        self.reset_parameters()
    
    def reset_parameters(self):
        #nn.init.zeros_(self.weight)
        nn.init.xavier_uniform_(self.rel_weight)
        nn.init.xavier_uniform_(self.basic_matrix)

    def forward(self, 
                hidden_states,
                rel_tensor,
                attention_mask=None,
                debug=False,
        ):
        """
        hidden_states: (batch, seq_len, input_size)
        rel_tensor: (batch, num_label, seq_len, seq_len)

        matmul(x, w)
        x: [batch_size, 1, seq_len, h] => [batch_size, n_out, seq_len, h], stack n_out times
        w: [n_out, h, h]) => [batch_size, n_out, h, h], stack batch_size times
        output: [batch_size, n_out, seq_len, h]
        """
        batch_size, seq_len, _ = list(hidden_states.size())
        device = hidden_states.device
        # (batch, num_basic, seq_len, output_size)
        #b_hidden_states = torch.matmul(hidden_states.unsqueeze(1), self.basic_matrix.unsqueeze(0))
        # (num_label, num_basic, seq_len, output_size)
        #weight = self.rel_weight.unsqueeze(-1).unsqueeze(-1).expand([-1,-1,seq_len,self.output_size])
        # (1, num_label, num_basic, seq_len, output_size) * (batch, 1, num_basic, seq_len, output_size)
        # => (batch, num_label, num_basic, seq_len, output_size)
        #b_r_hidden_states = weight.unsqueeze(0) * b_hidden_states.unsqueeze(1)
        # (batch, num_label, seq_len, output_size)
        #r_hidden_states = b_r_hidden_states.sum(2)
        # (num_label, num_basic, input_size, output_size)
        weight = self.rel_weight.unsqueeze(-1).unsqueeze(-1).expand([-1,-1,self.input_size,self.output_size])
        # (num_label, num_basic, input_size, output_size)
        rel_matrix = self.basic_matrix.unsqueeze(0) * weight
        # (num_label, input_size, output_size)
        rel_matrix = rel_matrix.sum(1)
        # (batch, 1, seq_len, input_size) x (1, num_label, input_size, output_size)
        # => (batch, num_label, seq_len, output_size)
        r_hidden_states = torch.matmul(hidden_states.unsqueeze(1), rel_matrix.unsqueeze(0))

        # (batch, num_label, seq_len, output_size)
        r_context_layer = torch.matmul(rel_tensor.float(), r_hidden_states)

        # (batch, num_label, seq_len)
        num_neighbors = rel_tensor.sum(-1)
        ones = torch.ones_like(num_neighbors, device=r_context_layer.device)
        num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
        if debug:
            print ("num_neighbors:\n", num_neighbors)
        # divide by the number of neighbors
        # (batch, num_label, seq_len, output_size)
        r_context_layer = r_context_layer / num_neighbors.unsqueeze(-1)
        # (batch, seq_len, output_size)
        context_layer = r_context_layer.sum(1)

        return context_layer
