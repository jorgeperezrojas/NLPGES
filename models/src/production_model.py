import torch
from models.src.ges_data_utils import IntToOneHotVectorConverter

class ProdGESModel( ):
    def __init__(self, model, vocabulary, tokenizer, especialidad_voc, edad_limits, class_labels, binary=False, unk_symbol='<unk>', device='cpu'):
        self.model = model
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.especialidad_voc = especialidad_voc
        self.class_labels = class_labels
        self.binary = binary
        self.unk_symbol = unk_symbol
        self.device = device        
        self.converter = IntToOneHotVectorConverter(edad_limits)

    def predict_class(self, raw_text_input, especialidad, edad):
        probs = self.predict_probabilities(raw_text_input, especialidad, edad)
        prediction = max(range(len(probs)), key=lambda x: probs[x])
        return prediction

    def predict(self, raw_text_input, especialidad, edad):
        prediction = self.predict_class(raw_text_input, especialidad, edad)
        return self.class_labels[prediction]

    def predict_labels_and_probabilities(self, raw_text_input, especialidad, edad):
        probs = self.predict_probabilities(raw_text_input, especialidad, edad)
        return list(zip(self.class_labels,probs))

    def predict_probabilities(self, raw_text_input, especialidad, edad):
        X_text, X_especialidad, X_edad, L = self._create_batch_from_raw_input(raw_text_input, especialidad, edad)
        X_text, X_especialidad, X_edad = X_text.to(self.device), X_especialidad.to(self.device), X_edad.to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        Y_logit = self.model((X_text, X_especialidad, X_edad, L))
        if self.binary:
            Y_prob = torch.sigmoid(Y_logit)
            Y_class = Y_prob.ge(0.5).float()
            Y_class = int(Y_class[0][0])
            prob_val = float(Y_prob[0][0])
            Y_prob = [1-prob_val, prob_val]
        else:
            Y_prob = torch.nn.functional.softmax(Y_logit, dim=1)
            Y_class = torch.argmax(Y_prob, dim=1) 
            Y_prob = [float(val) for val in Y_prob[0]]
            Y_class = int(Y_class[0])
        return Y_prob

    def _create_batch_from_raw_input(self, raw_text_input, especialidad, edad):
        tokens = self.tokenizer.tokenize_with_voc(raw_text_input, self.vocabulary, unk=self.unk_symbol)
        x_indexes = [self.vocabulary[word] for word in tokens]
        x_tensor = torch.tensor(x_indexes)
        x_seq_tensor = x_tensor.view(-1,1) # N is the first dimension
        x_especialidad = torch.tensor(self.especialidad_voc[especialidad]).view(1)
        x_edad = torch.tensor(self.converter.to_index(edad)).view(1)
        batch = (x_seq_tensor,x_especialidad,x_edad,[len(x_tensor)])
        return batch