import torch
from models.src.rnnattn import RNNAttnModel

class GESAttnModel(RNNAttnModel):
    def __init__(self,
        vocab_size,
        especialidad_voc_size,
        edad_limits_size,
        embedding_size = 50,
        especialidad_embedding_size = 20,
        edad_embedding_size = 20,
        input_dropout = 0,
        num_rnn_layers = 1,
        rnn_hidden_size = 50,
        bidirectional = False,
        lstm_dropout = 0,
        num_attn_heads = 2,
        fc_hidden_size = 30,
        fc_dropout = 0,
        classes = 2
    ):
        super(GESAttnModel, self).__init__(
                    vocab_size,
                    embedding_size = embedding_size,
                    input_dropout = input_dropout,
                    num_layers = num_rnn_layers,
                    rnn_hidden_size = rnn_hidden_size,
                    bidirectional = bidirectional,
                    lstm_dropout = lstm_dropout,
                    aggregation_type = 'attn',
                    num_attn_heads = num_attn_heads,
                    fc_hidden_size = fc_hidden_size,
                    fc_dropout = fc_dropout,
                    classes = classes)

        self.especialidad_emb_layer = torch.nn.Embedding(especialidad_voc_size, especialidad_embedding_size) 
        self.edad_emb_layer = torch.nn.Embedding(edad_limits_size, edad_embedding_size)

        directions = 2 if bidirectional else 1
        aggregation_size = directions * rnn_hidden_size

        # new layer to combine the general query and the especialidad embedding
        self.context_layer = torch.nn.Linear(self.query.size()[0] + especialidad_embedding_size, aggregation_size)
        # redefine the fc_layer
        self.fc_1 = torch.nn.Linear(aggregation_size + edad_embedding_size, fc_hidden_size)


    def reset_parameters(self):
        super(GESAttnModel, self).reset_parameters()
        # TODO: reset self.especialidad_emb_layer, self.edad_emb_layer, self.context_layer, self.fc_1
        pass
        

    def forward(self, X):
        # assume input is of the form X = (X_text, X_especialidad, X_edad, lengths)
        # - X_text is a tensor obtained from a padded sequence of text (in the form of word sequences), 
        # with time-step first (time,batch,features)
        # - X_especialidad is the index of an embedding for especialidad
        # - X_edad is the index of an embedding for edad
        # - lengths is a tensor containing the lengths of every sequence in X_text

        X_text, X_especialidad, X_edad, lengths = X
        batch_size = len(lengths)

        # embeddings + rnn for X_text
        H = self.emb_layer(X_text)
        H = self.input_dropout(H)
        H = torch.nn.utils.rnn.pack_padded_sequence(H, lengths)
        O, (_, _) = self.lstm(H)
        O, _ = torch.nn.utils.rnn.pad_packed_sequence(O)

        # compute context vectors for attention from self.query and X_especialidad
        E = self.especialidad_emb_layer(X_especialidad)
        C = self.query.repeat(batch_size, 1)
        C = torch.cat((C, E), 1)
        C = self.context_layer(C)
        C = torch.nn.functional.relu(C)

        # use context vector to compute the attention
        H = self.attention(C, O)

        # use the attention plus X_edad to produce the final output
        E = self.edad_emb_layer(X_edad)
        H = torch.cat((H, E), 1)
        H = self.fc_1_dropout(H)
        H = self.fc_1(H)
        H = torch.nn.functional.relu(H)
        H = self.fc_2_dropout(H)
        H = self.fc_2(H)
        return H 
        
