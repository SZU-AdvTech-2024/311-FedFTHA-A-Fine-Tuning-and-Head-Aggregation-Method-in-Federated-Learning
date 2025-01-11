import time
import torch
import numpy as np

from system.flcore.clients.clientbase import Client


class ClientRep(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(self.plocal_epochs):

            for i, (x,y) in enumerate(trainloader):
                if type[x] == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1*np.abs(np.random.rand()))

                output = self.model(x)
                self.optimizer.zero_grad()
                loss_val = self.loss(output, y)
                loss_val.backward()
                self.optimizer.step()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(self.local_epochs):
            for i, (x,y) in enumerate(trainloader):
                if type[x] == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1*np.abs(np.random.rand()))

                output = self.model(x)
                self.optimizer_per.zero_grad()
                loss_val = self.loss(output, y)
                loss_val.backward()
                self.optimizer_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
            for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()