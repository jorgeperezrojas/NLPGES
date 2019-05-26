import torch
from models.src.rnnattn import RNNAttnModel
import ipdb

class GESAttnModel(RNNAttnModel):
    def __init__(self,
        vocab_size,
        especialidad_voc_size,
        edad_limits_size,
        embedding_size = 50,
        especialidad_embedding_size = 20,
        edad_embedding_size = 40,
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
        
    def _rnn(self, X):
        X_text, _, _, lengths = X

        # embeddings + rnn for X_text
        H = self.emb_layer(X_text)
        H = self.input_dropout(H)
        H = torch.nn.utils.rnn.pack_padded_sequence(H, lengths)
        O, (_, _) = self.lstm(H)
        O, _ = torch.nn.utils.rnn.pad_packed_sequence(O)
        return O

    def _context(self, X, O):
        # for now we just ignore O
        _, X_especialidad, _, lengths = X
        batch_size = len(lengths)

        # compute context vectors for attention from self.query and X_especialidad
        E = self.especialidad_emb_layer(X_especialidad)
        C = self.query.repeat(batch_size, 1)
        C = torch.cat((C, E), 1)
        C = self.context_layer(C)
        C = torch.nn.functional.relu(C)
        return C

    def _edad_embedding(self, X):
        _, _, X_edad, _ = X

        # compute an embedding for X_edad
        E = self.edad_emb_layer(X_edad)
        E = torch.nn.functional.relu(E)
        return E

    def _output(self, A, E):
        H = torch.cat((A, E), 1)
        H = self.fc_1_dropout(H)
        H = self.fc_1(H)
        H = torch.nn.functional.relu(H)
        H = self.fc_2_dropout(H)
        H = self.fc_2(H)
        return H        

    def forward(self, X):
        # Assumes input is of the form X = (X_text, X_especialidad, X_edad, lengths)
        # - X_text is a tensor obtained from a padded sequence of text (in the form of word sequences), 
        # with time-step first (time,batch,features)
        # - X_especialidad is the index of an embedding for especialidad
        # - X_edad is the index of an embedding for edad
        # - lengths is a tensor containing the lengths of every sequence in X_text

        # rnn
        O = self._rnn(X)
        # context
        C = self._context(X, O)
        # use context vector to compute the attention
        A = self._attention(C, O)
        # compute an embedding for the edad input
        E = self._edad_embedding(X)
        # use the attention plus the edad embedding to produce the final output
        H = self._output(A, E)
        return H 
        
class GESOrderedAttnModel(GESAttnModel):
    def __init__(self,
        vocab_size,
        especialidad_voc_size,
        edad_limits_size,
        **kwargs):

        super(GESOrderedAttnModel, self).__init__(
                    vocab_size,
                    especialidad_voc_size,
                    edad_limits_size,
                    **kwargs)

        ## get the edad embedding size
        edad_embedding_size = self.edad_emb_layer.embedding_dim
        ## delete the embedding layer for X_edad
        del(self.edad_emb_layer)

        ## add new parameters for ordered edad embeddings
        self.left_edad_embedding = torch.nn.Linear(edad_limits_size,edad_embedding_size, bias=False)
        self.right_edad_embedding = torch.nn.Linear(edad_limits_size,edad_embedding_size, bias=False)
        self.edad_emb_layer = torch.nn.Linear(3*edad_embedding_size,edad_embedding_size, bias=False)

        ## for later use
        self.edad_limits_size = edad_limits_size


    def _edad_embedding(self, X):
        # ipdb.set_trace()
        _, _, X_edad, _ = X

        # ordered layer for X_edad
        # create the identity
        I = torch.eye(self.edad_limits_size)
        I = I.to(X_edad.device)
        C = I[X_edad]
        R = C.cumsum(1)
        L = C.new_ones(self.edad_limits_size) - R + C

        # compute the center embedding for X_edad
        C = self.left_edad_embedding(C) + self.right_edad_embedding(C)

        # compute left and right embeddings for X_edad
        L = self.left_edad_embedding(L)
        R = self.right_edad_embedding(R)
        
        # compute the final embedding
        E = torch.cat((L,C,R), 1)
        E = self.edad_emb_layer(E)
        E = torch.nn.functional.relu(E)
        return E
