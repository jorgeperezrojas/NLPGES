import torch

class RNNAttnModel(torch.nn.Module):
    def __init__(self,
        vocab_size,
        embedding_size = 50,
        input_dropout = 0,
        num_layers = 1,
        rnn_hidden_size = 50,
        bidirectional = False,
        lstm_dropout = 0,
        aggregation_type = 'last',
        num_attn_heads = 1,
        fc_hidden_size = 30,
        fc_dropout = 0,
        classes = 2
    ):
        super(RNNAttnModel, self).__init__()        
        self.emb_layer = torch.nn.Embedding(vocab_size, embedding_size)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.lstm = torch.nn.LSTM(
            input_size = embedding_size, hidden_size = rnn_hidden_size,
            num_layers = num_layers, dropout = lstm_dropout, bidirectional = bidirectional)       
        directions = 2 if bidirectional else 1
        assert aggregation_type in ['last','max','avg','attn']
        self.aggregation_type = aggregation_type
        aggregation_size = directions * rnn_hidden_size
        self.fc_1_dropout = torch.nn.Dropout(fc_dropout)
        self.fc_1 = torch.nn.Linear(aggregation_size, fc_hidden_size)
        self.fc_2_dropout = torch.nn.Dropout(fc_dropout)
        self.fc_2 = torch.nn.Linear(fc_hidden_size, classes)
        if self.aggregation_type == 'attn':
            assert aggregation_size % num_attn_heads == 0, f'Number of heads ({num_attn_heads}) should divide representation size ({aggregation_size})'
            self.head_size = int(aggregation_size / num_attn_heads)
            self.num_attn_heads = num_attn_heads
            self.query = torch.nn.Parameter(torch.ones(aggregation_size)) # decide the best way of initialization
            self.Wquery = torch.nn.Parameter(torch.randn(num_attn_heads,aggregation_size,self.head_size))
            self.Wkey = torch.nn.Parameter(torch.randn(num_attn_heads,aggregation_size,self.head_size))
            self.Wvalue = torch.nn.Parameter(torch.randn(num_attn_heads,aggregation_size,self.head_size))
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: define a suitable initialization, specially for attention parameters
        pass

    def processes_lstm_output(self, O, H, C):
        # C is not used by now
        directions = 2 if self.lstm.bidirectional else 1
        if self.aggregation_type == 'max' or self.aggregation_type == 'avg':
            O, lengths = torch.nn.utils.rnn.pad_packed_sequence(O)
            if self.aggregation_type == 'max':
                O, _ = torch.max(O, 0) 
            elif self.aggregation_type == 'avg':
                O = torch.sum(O,0)
                O = O / lengths.float().view(-1,1).to(O.device)
            return O
        if self.aggregation_type == 'last': 
            # this code is to obtain the last state of the last layer
            # it is a pain when several directions/layers are used in rnns
            layers = self.lstm.num_layers
            hidden_size = self.lstm.hidden_size
            H = H.view(layers, directions, -1, hidden_size)
            H = H[-1].transpose(0,1).contiguous()
            H = H.view(-1, directions * hidden_size)
            return H
        else:
            pass
        
    def _attention(self, context_vector, input_vectors):
        # TODO: optimize Q computation when no batch is provided in the context vector
        # TODO: try to change the einsum computations by tensordot or bmm when possible
        if len(context_vector.size()) == 1: 
            # if there is no batch dimension, add it
            batch_size = input_vectors.size()[1]
            context_vector = context_vector.view(1,-1).repeat(batch_size,1)
        # t:time, b:batch, f:features, n:attn-heads, h:attn-size
        Q = torch.einsum('bf,nfh->bnh', context_vector, self.Wquery)
        K = torch.einsum('tbf,nfh->tbnh', input_vectors, self.Wkey)
        V = torch.einsum('tbf,nfh->tbnh', input_vectors, self.Wvalue)
        scale = Q.new_full((1,),self.head_size)
        pre_scores = torch.einsum('tbnh,bnh->tbn',K,Q) / scale
        scores = torch.nn.functional.softmax(pre_scores, dim=0)
        attn_heads = torch.einsum('tbn,tbnh->bnh', scores, V)
        A = attn_heads.view(-1,self.num_attn_heads*self.head_size)
        return A
               
    def forward(self, X_plus_lengths):
        # asume input X_plus_lengths is a pair (X,lengths) where:
        # - X is a tensor obtained from a padded sequence, with time-step first (time,batch,features)
        # - lengths is a tensor containing the lengths of every sequence
        X, lengths = X_plus_lengths
        H = self.emb_layer(X)
        H = self.input_dropout(H)
        H = torch.nn.utils.rnn.pack_padded_sequence(H, lengths)
        O, (H, C) = self.lstm(H)
        if self.aggregation_type != 'attn':
            H = self.processes_lstm_output(O, H, C)
        else:
            O, _ = torch.nn.utils.rnn.pad_packed_sequence(O)
            q = self.query
            H = self._attention(q, O)
        H = self.fc_1_dropout(H)
        H = self.fc_1(H)
        H = torch.nn.functional.relu(H)
        H = self.fc_2_dropout(H)
        H = self.fc_2(H)
        return H 