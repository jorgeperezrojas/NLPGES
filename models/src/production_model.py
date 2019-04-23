import torch

class ProdTextModel():
    def __init__(self, model, vocabulary, tokenizer, class_labels, binary=False, unk_symbol='<unk>', device='cpu'):
        self.model = model
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.class_labels = class_labels
        self.binary = binary
        self.unk_symbol = unk_symbol
        self.device = device

    def predict_class(self, raw_text_input):
        probs = self.predict_probabilities(raw_text_input)
        prediction = max(range(len(probs)), key=lambda x: probs[x])
        return prediction

    def predict(self, raw_text_input):
        prediction = self.predict_class(raw_text_input)
        return self.class_labels[prediction]

    def predict_labels_and_probabilities(self, raw_text_input):
        probs = self.predict_probabilities(raw_text_input)
        return list(zip(self.class_labels,probs))

    def predict_probabilities(self, raw_text_input):
        X, L = self._create_batch_from_raw_text(raw_text_input)
        X = X.to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        Y_logit = self.model((X,L))
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
        #return list(zip(self.classes,Y_prob))

    def _create_batch_from_raw_text(self, raw_text_input):
        tokens = self.tokenizer.tokenize_with_voc(raw_text_input, self.vocabulary, unk=self.unk_symbol)
        x_indexes = [self.vocabulary[word] for word in tokens]
        x_tensor = torch.tensor(x_indexes)
        x_seq_tensor = x_tensor.view(-1,1)
        batch = (x_seq_tensor,[len(x_tensor)])
        return batch