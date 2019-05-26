import torch
import time
import itertools
import sys
from datetime import datetime


def grid_from_hyper_params_options(hps_prefix='__hps__', hparameter_dict={}):
    '''
    Generator to create a gird over hyperparameter options. Every key in hparameter_dict
    starting with hps_prefix is considered to be a set of options for that hyperparameter.
    For every possible combination of options, the generator produces a set of hyperparameters.
    '''
    base_dict = {}
    lists_dict = {}
    for key in hparameter_dict:
        if key.startswith(hps_prefix):
            real_key = key[len(hps_prefix):]
            lists_dict[real_key] = hparameter_dict[key]
        else:
            base_dict[key] = hparameter_dict[key]

    key_list = [k for k in lists_dict]
    hp_lists = [lists_dict[k] for k in key_list]
    for prod in itertools.product(*hp_lists):
        dict_to_add = {k:prod[i] for i,k in enumerate(key_list)}
        output = {**base_dict, **dict_to_add}
        yield(output)


class MyTrainer():

    def __init__(self, model, optimizer, loss_fn, device='cuda:0'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, train_data, val_data, epochs, patience=5, padded_data=True, 
        prepare_batch_fn=None, verbose=True, device=None, val_device=None, seed=333):

        if device == None:
            device = self.device
        if val_device == None:
            val_device = self.device

        torch.manual_seed(seed)

        self.model = self.model.to(device)
        max_dev_acc, max_epoch = MySearchTrainer.train_specific_model(
                self.model, self.optimizer, self.loss_fn, train_data, val_data, epochs,
                device, val_device, padded_data, prepare_batch_fn=prepare_batch_fn, 
                verbose=verbose, patience=patience, output_file_details=None)

        print_info = f'max dev_acc:{100*max_dev_acc:02.2f} at epoch:{max_epoch}\n'
        print(print_info)


class MySearchTrainer():
    # TODO: sacar las funcionalidades de train_specific_model y eval
    def __init__(self, 
                 model_class, optimizer_class, loss_fn,
                 train_data, val_data, padded_data=True, prepare_batch_fn=None,
                 model_params={}, optimizer_params={},
                 hps_prefix='__hps__',
                 epochs=2,
                 device='cuda:0',
                 patience=5,
                 seed=333,
                 output_file_prefix='models/results/out_train'):
        
        self.model_class = model_class
        self.model_params_generator = grid_from_hyper_params_options(hps_prefix,model_params)
        
        self.optimizer_class = optimizer_class
        self.optimizer_params_generator = grid_from_hyper_params_options(hps_prefix,optimizer_params)
        
        self.loss_fn = loss_fn
        
        self.train_data = train_data
        self.val_data = val_data
        self.padded_data = padded_data
        self.prepare_batch_fn = prepare_batch_fn
        
        self.epochs = epochs
        self.device = device
        self.patience = patience

        self.output_file_prefix = output_file_prefix
        self.seed = seed
        self.best_model = None
        self.best_dev_acc = 0.0
        self.m_ps = None
    
    def _generate_file_names(self):
        now_str = datetime.now().strftime('%Y%m%d%H%M')
        self.output_filenames = {}
        output_file_types = ['details','models','best']
        for file_type in output_file_types:
            self.output_filenames[file_type] = self.output_file_prefix + '_' + file_type + '_' + now_str + '.txt'

    def train_models(self):
        self._generate_file_names()
        print('Writting on files ' + str(list(self.output_filenames.values())))

        for m_ps, op_ps in itertools.product(self.model_params_generator, self.optimizer_params_generator):
            torch.manual_seed(self.seed)
            model = self.model_class(**m_ps)
            model = model.to(self.device)
            optimizer = self.optimizer_class(model.parameters(), **op_ps)
            
            info = 'Training model ' + str(self.model_class) + ' with hyperparams ' + str(m_ps) +\
                    ' and optimizer ' + str(self.optimizer_class) + ' with hyperparams ' + str(op_ps)
            print(info)
            with open(self.output_filenames['details'],'a') as outfile_details, \
                    open(self.output_filenames['models'],'a') as outfile_models:
                outfile_details.write(info)
                outfile_details.write('\n')
                outfile_models.write(info)
                outfile_models.write('\n')

            max_dev_acc, max_epoch = self.train_specific_model(
                model, optimizer, self.loss_fn, self.train_data, self.val_data, self.epochs,
                self.device, self.device, self.padded_data, prepare_batch_fn=self.prepare_batch_fn, verbose=True, patience=self.patience,
                output_file_details=self.output_filenames['details']
            )

            print_info = f'max dev_acc:{100*max_dev_acc:02.2f} at epoch:{max_epoch}\n'
            print(print_info)

            with open(self.output_filenames['best'],'a') as outfile_best:
                info = f'{100*max_dev_acc:02.2f},{max_epoch}\n'
                outfile_best.write(info)

            self.__save_best_model(model, max_dev_acc, m_ps)

        print()
        print('best dev_acc:', self.best_dev_acc)
        print('best model params:', self.m_ps)

    def __save_best_model(self, model, dev_acc, m_ps):
        # TODO: implement this function! for now it is only the identity
        if dev_acc > self.best_dev_acc:
            self.best_dev_acc = dev_acc
            self.best_model = model
            self.m_ps = m_ps


    def eval(self, data, padded_data, device, prepare_batch_fn=None):
        return MySearchTrainer.eval_specific_model(self.best_model, self.loss_fn, data, padded_data, 
            device, prepare_batch_fn=prepare_batch_fn)


    def predict(self, X, padded_data, device, prepare_batch_fn=None):
        model = self.best_model
        model.eval()
        if padded_data:
            (X, lengths) = X
            X = X.to(device)
            X = (X, lengths)
        else:
             X = X.to(device)
        Y_pred = model(X)
        Y_class, Y_prob = MySearchTrainer.logit_to_class(Y_pred)
        return Y_class, Y_prob

    @staticmethod
    def logit_to_class(Y_logit):
        # TODO: decide if sigmoid or softmax depending on a parameter not the size of Y_logit
        if Y_logit.size()[1] == 1:
            Y_prob = torch.sigmoid(Y_logit)
            Y_class = Y_prob.ge(0.5).float()
        else:
            Y_prob = torch.nn.functional.softmax(Y_logit, dim=1)
            Y_class = torch.argmax(Y_prob, dim=1) 
        return Y_class, Y_prob   
    
    @staticmethod
    def eval_specific_model(model, loss_fn, data, padded_data, device, prepare_batch_fn=None):
        model = model.to(device)
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_examples = 0
        for batch in data:
            if prepare_batch_fn == None:
                (X, Y) = MySearchTrainer._prepare_data_from_batch(batch, padded_data, device)
            else:
                (X, Y) = prepare_batch_fn(batch, device)
            Y_pred = model(X)
            Y_class, Y_prob = MySearchTrainer.logit_to_class(Y_pred)
            running_loss += loss_fn(Y_pred, Y).item()
            running_acc += torch.sum(Y == Y_class).item() 
            total_examples += len(Y)
        loss = running_loss / total_examples
        acc = running_acc / total_examples
        return loss, acc
                    
    @staticmethod
    def train_specific_model(
            model, optimizer, loss_fn, train_data, dev_data, epochs,  
            device, eval_device, padded_data, prepare_batch_fn=None, 
            verbose=False, patience=1, output_file_details=None):
        # assume that (all) model parameters and optimizer are in the same device 
        # as the one passed as argument
        acc_not_improving = 0
        max_dev_acc = 0.0
        max_epoch = 0
        for epoch in range(epochs):
            epoch_init_time = time.time()
            running_loss = 0.0
            running_acc = 0.0
            total_examples = 0
            for i, batch in enumerate(train_data):
                model.train()
                optimizer.zero_grad()
                if prepare_batch_fn == None:
                    (X, Y) = MySearchTrainer._prepare_data_from_batch(batch, padded_data, device)
                else:
                    (X, Y) = prepare_batch_fn(batch, device)
                Y_pred = model(X)
                loss = loss_fn(Y_pred, Y)
                loss.backward()
                optimizer.step()
                Y_class, Y_prob = MySearchTrainer.logit_to_class(Y_pred)
                running_loss += loss.item()
                running_acc += torch.sum(Y == Y_class).item()
                total_examples += len(Y)
                partial_info = f'\rEpoch:{epoch+1:03} batch:{i+1}/{len(train_data)} ' +\
                    f'running_loss:{running_loss/total_examples:02.6f} ' +\
                    f'running_acc:{100*running_acc/total_examples:02.2f}%    '
                sys.stdout.write(partial_info)
            
            elapsed_time = (time.time() - epoch_init_time)
            train_loss = running_loss/total_examples
            train_acc = running_acc/total_examples
            dev_loss, dev_acc = MySearchTrainer.eval_specific_model(model, loss_fn, dev_data, padded_data, eval_device, prepare_batch_fn=prepare_batch_fn)
            out_info_iter = \
                f'\rEpoch:{epoch+1:03} in {elapsed_time:03.0f}s ' +\
                f'train_loss:{train_loss:02.6f}, train_acc:{100*train_acc:02.2f}% ' +\
                f'dev_loss:{dev_loss:02.6f}, dev_acc:{100*dev_acc:02.2f}%'
            print(out_info_iter)

            if output_file_details != None:
                with open(output_file_details,'a') as outfile:
                    outfile.write(out_info_iter)
                    outfile.write('\n')
            
            if dev_acc > max_dev_acc:
                acc_not_improving = 0
                max_dev_acc = dev_acc
                max_epoch = epoch+1
            else:
                acc_not_improving += 1
                if acc_not_improving >= patience:
                    print('Val_acc not improving for ' + str(patience) + ' iterations. Stopping.')
                    return (max_dev_acc, max_epoch)
                    
            # TODO: log train data
        return(max_dev_acc, max_epoch)
            
    @staticmethod
    def _prepare_data_from_batch(batch, padded_data, device):
        (X,Y) = batch        
        if padded_data:
            (X, lengths) = X
            X = X.to(device)
            X = (X, lengths)
        else:
             X = X.to(device)
        Y = Y.to(device)
        return (X,Y)
