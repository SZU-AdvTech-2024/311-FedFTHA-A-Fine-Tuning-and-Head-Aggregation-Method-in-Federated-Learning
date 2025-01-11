import time
import torch
import numpy as np

from system.flcore.clients.clientbase import Client


class ClientFTHA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(self.sync_local_epochs):

            for i, (x,y) in enumerate(trainloader):
                if type[x] == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1*np.abs(np.random.rand()))

                if self.sam:
                output = self.model(x)
                self.optimizer.zero_grad()
                loss_val = self.loss(output, y)
                epsilon = 0.05  # SAM扰动幅度，可以调整
                loss_val.backward()
                with torch.no_grad():
                    grads = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grads.append(param.grad.norm())
                    first_grad_norm = torch.norm(torch.stack(grads)) + 1e-14
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grad_norm = epsilon * param.grad / first_grad_norm
                            param.add_(grad_norm)
                            param._eps = grad_norm

                output = self.model(x)
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