import time

import numpy as np

from system.flcore.clients.clientbase import Client


class ClientAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

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
                self.optimizer.zero_grad()
                loss_val = self.loss(output, y)
                loss_val.backward()
                self.optimizer.step()