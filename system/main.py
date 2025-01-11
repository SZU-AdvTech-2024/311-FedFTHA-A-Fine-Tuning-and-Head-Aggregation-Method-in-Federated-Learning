import argparse
import copy
import logging
import os
import time
import warnings

import torch
import torchvision
from torch import nn

from system.flcore.servers.serverftha import FedFTHA
from system.flcore.servers.serverrep import FedRep
from system.flcore.trainmodel.alexnet import alexnet
from system.flcore.trainmodel.bilstm import BiLSTM_TextClassification
from system.flcore.trainmodel.mobilenet_v2 import mobilenet_v2
from system.flcore.trainmodel.models import Mclr_Logistic, DNN, CNNCifar_PFL, FedAvgCNN, Digit5CNN, LSTMNet, fastText, \
    TextCNN, AmazonMLP, HARCNN, BaseHeadSplit
from system.flcore.trainmodel.resnet import resnet18, resnet10
from system.flcore.trainmodel.transformer import TransformerModel
from system.utils.mem_utils import MemReporter

from flcore.servers.serveravg import FedAvg

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635  # 98635 for AG_News and 399198 for Sogou_News
max_len = 200
emb_dim = 32

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":  # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "dnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        elif model_str == "cnn":  # non-convex
            if "fmnist" in args.dataset:
                args.model = CNNCifar_PFL(args=args).to(args.device)
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Cifar100" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "resnet":
            # args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

            args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            feature_dim = list(args.model.fc.parameters())[0].shape[1]
            args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "resnet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                                   output_size=args.num_classes,
                                                   num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                                   attention_dropout=0,
                                                   embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2,
                                          num_classes=args.num_classes).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

    if args.algorithm == "FedAvg":
        # args.head = copy.deepcopy(args.model.fc)
        # args.model.fc = nn.Identity()
        # args.model = BaseHeadSplit(args.model, args.head)

        server = FedAvg(args, i)

    elif args.algorithm == "FedRep":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedRep(args, i)

    elif args.algorithm == "FedFTHA":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedFTHA(args, i)

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)  # -------------------设置-训练轮数
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,  # 本地训练轮数
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-m', "--model", type=str, default="resnet")  # -------------------------------选择-模型
    parser.add_argument('-algo', "--algorithm", type=str, default="FedFTHA")  # 选择-算法-204行处有相应的算法选择
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.1,
                        help="Ratio of clients per round")
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=10)
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")  # ----------------------选择-数据集
    parser.add_argument('-nb', "--num_classes", type=int, default=10)  # ----------------------------类别数
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")

    # FedFTHA
    parser.add_argument('-sync', "--sync_local_epochs", type=int, default=5,  # 本地训练轮数
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-sam', "--is_sam", type=bool, default=False, help="Whether the model uses SAM")

    parser.add_argument('-move', "--is_move", type=bool, default=False, help="Whether the model uses MOVE AVERAGE")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
        args.device = "cpu"
    else:
        print("\ncuda is available.\n")

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    run(args)