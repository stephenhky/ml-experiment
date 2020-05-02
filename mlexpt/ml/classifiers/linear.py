
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core import ExperimentalClassifier


class TorchLogisticRegression(nn.Module):
    def __init__(self, nbinputs, nboutputs, device='cpu'):
        super(TorchLogisticRegression, self).__init__()
        self.nbinputs = nbinputs
        self.nboutputs = nboutputs
        self.device = torch.device(device)

        self.linearblock = nn.Linear(nbinputs, nboutputs).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        logodds = self.linearblock(x)
        # y = self.sigmoid(logodds)    # sigmoid or softmax in cost function
        return logodds


class MulticlassLogisticRegression(ExperimentalClassifier):
    def __init__(self, device=torch.device('cpu'), nb_epoch=100, batch_size=10000):
        self.device = device
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

    def fit(self, x, y):
        # x.shape = (m, n)
        # y.shape = (m, nboutputs)
        dataloader = DataLoader(np.concatenate([x, y], axis=1),
                                batch_size=min(x.shape[0], self.batch_size))

        self.logregs = TorchLogisticRegression(x.shape[1], y.shape[1], self.device)
        print('Logistic regression trained on: '+self.logregs.device.type)

        input_dim = x.shape[1]
        nbclasses = y.shape[1]
        if nbclasses > 1:
            criterion = nn.CrossEntropyLoss().to(self.logregs.device)
            self.activation_function = nn.Sigmoid()
        else:
            criterion = nn.BCEWithLogitsLoss().to(self.logregs.device)
            self.activation_function = nn.Softmax()
        optimizer = torch.optim.Adam(self.logregs.parameters(), lr=0.01)

        for _ in tqdm(range(self.nb_epoch)):
            for data in dataloader:
                optimizer.zero_grad()
                X = data[:, :input_dim].type(torch.FloatTensor).to(self.logregs.device)
                Y = data[:, input_dim:].type(torch.FloatTensor).to(self.logregs.device)

                pred_Y = self.logregs(X)

                if nbclasses > 1:
                    target_indices = torch.max(Y.to(self.logregs.device), 1)[1]
                    loss = criterion(pred_Y, target_indices)
                else:
                    loss = criterion(pred_Y, Y.to(self.logregs.device))

                loss.backward()
                optimizer.step()

    def fit_batch(self, numerically_batched_dataset):
        # x.shape = (m, n)
        # y.shape = (m, nboutputs)
        input_dim = numerically_batched_dataset.nbinputs
        nbclasses = numerically_batched_dataset.nboutputs

        self.logregs = TorchLogisticRegression(input_dim, nbclasses, self.device)
        print('Logistic regression trained on: '+self.logregs.device.type)

        if nbclasses > 1:
            criterion = nn.CrossEntropyLoss().to(self.logregs.device)
            self.activation_function = nn.Sigmoid()
        else:
            criterion = nn.BCEWithLogitsLoss().to(self.logregs.device)
            self.activation_function = nn.Softmax()
        optimizer = torch.optim.Adam(self.logregs.parameters(), lr=0.01)

        for _ in tqdm(range(self.nb_epoch)):
            for fileid in range(numerically_batched_dataset.nbbatches):
                optimizer.zero_grad()
                X, Y = numerically_batched_dataset.get_batch(fileid)
                X = X.to(self.logregs.device)
                Y = Y.to(self.logregs.device)
                pred_Y = self.logregs(X)

                if nbclasses > 1:
                    target_indices = torch.max(Y, 1)[1]
                    loss = criterion(pred_Y, target_indices)
                else:
                    loss = criterion(pred_Y, Y)

                loss.backward()
                optimizer.step()

    def predict_proba(self, x):
        y = self.activation_function(self.logregs(torch.FloatTensor(x)))
        return y.detach().cpu().numpy()

    def predict_proba_batch(self, dataset):
        predicted_Y = None
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        for data in dataloader:
            x, _ = data
            new_pred_y = self.activation_function(self.logregs(x))
            if predicted_Y is None:
                predicted_Y = new_pred_y.detach().cpu().numpy()
            else:
                predicted_Y = np.append(predicted_Y, new_pred_y.detach().cpu().numpy(), axis=0)
        return predicted_Y

    def persist(self, path):
        torch.save(self.logregs.state_dict(), path)

    @classmethod
    def load(cls, modelpath, device='cpu'):
        state_dict = torch.load(modelpath)
        nboutputs, nbinputs = state_dict['linearblock.weight'].shape
        model = cls(device=torch.device(device))
        model.logregs = TorchLogisticRegression(nbinputs, nboutputs)
        model.logregs.load_state_dict(state_dict)
        model.activation_function = nn.Sigmoid() if nboutputs > 1 else nn.Softmax()
        return model

