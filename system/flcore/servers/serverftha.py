import copy
from datetime import time
import random

import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from system.flcore.clients.clientftha import ClientFTHA
from system.flcore.servers.serverbase import Server
from system.utils.data_utils import read_all_clients_data


class FedFTHA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(ClientFTHA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.selected_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personal model")
                self.evaluate()
                print("\nEvaluate personal model")
                self.evaluate_global()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.move:
                self.average_parameters()
            else:
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        self.save_results()

    def evaluate_global(self, acc=None, loss=None):
        self.global_model.eval()

        test_data = read_all_clients_data(self.dataset, num_clients=self.num_clients, is_train=False)
        testloaderfull = DataLoader(test_data, self.batch_size, drop_last=False, shuffle=True)

        test_acc, test_num, y_prob, y_true = 0, 0, [], []

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes + 1 if self.num_classes == 2 else self.num_classes
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        test_acc = test_acc / test_num

        print(f"Averaged Test Accuracy: {test_acc:.4f}")
        print(f"Averaged Test AUC: {test_auc:.4f}")

    def move_aggreate_average(self):
        assert (len(self.uploaded_models) > 0)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.average_parameters(w, client_model)

    def average_parameters(self, w, client_model):
        b = 0.1
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data = (1 - b) * server_param.data + b * client_param.data



