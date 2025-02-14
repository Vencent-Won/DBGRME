#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse


def RS_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", default=0, type=int, help="gpu device id")
    parser.add_argument("--cuda", action='store_true', default=True, help="Enable Cuda")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--exp_id", default="", type=str, help="Experiment ID")
    parser.add_argument("--exp_name", default="run", type=str, help="Experiment name")
    parser.add_argument("--dump_path", default="Log/", type=str, help="Experiment dump path")
    parser.add_argument("--model_path", default="pretrain_models", type=str, help="Experiment Path")
    parser.add_argument("--logger_path", default=None, type=str, help="Experiment Path")

    parser.add_argument("--datadir", default='./data/', type=str, help="the path of dataset")
    parser.add_argument("--dataset", default='ml-1m', type=str, help="dataset name")


    parser.add_argument("--model_name", default="lgn", type=str, help="[bpr, ncf, gcmc,  ngcf, lgn]")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="the weight decay of optimizer")
    parser.add_argument("--hidden_dim_benign", default=64, type=int, help="the hidden dim of benign model")
    parser.add_argument("--num_layers_benign", default=3, type=int, help="the number of layers of benign model")


    parser.add_argument("--top_k", default=50, type=int, help="the top-k of evaluation")
    parser.add_argument("--epoch", default=1000, type=int, help="training epoch")
    parser.add_argument("--eval_gap", default=10, type=int, help="eval per n epoch")
    parser.add_argument("--lr_benign", default=1e-3, type=float, help="learning rate of benign model")
    parser.add_argument("--batch_size", default=8192, type=int, help="the batch size of benign training")
    parser.add_argument('--dropout_rate', default=0., type=float, help="dropout rate")
    parser.add_argument("--test_batch_size", default=4000, type=int, help="the batch size of test")

    args = parser.parse_args()

    return args



def RSMSA_args_parser():
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--gpu", default=0, type=int, help="gpu device id")
    parser.add_argument("--cuda", action='store_true', default=True, help="Enable Cuda")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--exp_id", default="", type=str, help="Experiment ID")
    parser.add_argument("--exp_name", default="run", type=str, help="Experiment name")
    parser.add_argument("--dump_path", default="Log/", type=str, help="Experiment dump path")
    parser.add_argument("--logger_path", default=None, type=str, help="Experiment Path")

    # benign model setting
    parser.add_argument("--datadir", default='./data/', type=str, help="the path of dataset")
    parser.add_argument("--dataset", default='ml-100k', type=str, help="dataset name")
    parser.add_argument("--model_path", default="pretrain_models", type=str, help="Experiment Path")
    parser.add_argument("--regs_decay", default=1e-4, type=float, help='bpr efficient')
    parser.add_argument("--victim_model", default="ngcf", type=str, help="[bpr, ncf, gcmc, ngcf, lgn]")
    parser.add_argument("--hidden_dim_benign", default=64, type=int, help="the hidden dim of benign model")
    parser.add_argument("--num_layers_benign", default=3, type=int, help="the number of layers of benign model")

    # clone model setting
    parser.add_argument("--lr_clone", default=5e-4, type=float)
    parser.add_argument("--iter_clone", default=1, type=int)
    parser.add_argument("--clone_first", action='store_true', default=False, help="1: clone first, 0: generate first")
    parser.add_argument("--fit_top_k", default=100, type=int, help="the top-k of victim feedback")
    parser.add_argument('--dropout', default=0.1, type=float, help="margin of ranking loss")
    parser.add_argument('--alpha_norm', default=1, type=float, help="morm loss rate")
    parser.add_argument("--hidden_dim_clone", default=128, type=int, help="the hidden dim of clone model")
    parser.add_argument("--num_layers_clone", default=3, type=int, help="the number of layers of clone model")

    # generate setting
    parser.add_argument("--lr_generator", default=1e-4, type=float)
    parser.add_argument("--iter_generator", default=1, type=int)
    parser.add_argument("--gen_top_k", default=50, type=int, help="the top-k of interaction")
    parser.add_argument("--num_fakers", default=300, type=int, help="a times the number of generated users")
    parser.add_argument("--in_dim_generator", default=128, type=int, help="the input dim of generator")
    parser.add_argument("--hidden_dim_generator", default=256, type=int, help="the hidden1 dim of generator")
    parser.add_argument("--out_dim_generator", default=128, type=int, help="the hidden2 dim of generator")
    parser.add_argument("--num_layers_generator", default=4, type=int, help="the number of layers of generator")

    # optimize setting
    parser.add_argument("--eval_gap", default=300, type=int, help="eval per n epoch")
    parser.add_argument("--eval_top_k", default=50, type=int, help="the top-k of evaluation")
    parser.add_argument("--query_budget", default=200000, type=int, help="")
    parser.add_argument("--test_batch_size", default=4096, type=int, help="the batch size of test")
    parser.add_argument("--train_bpr_batch_size", default=4096, type=int)


    args = parser.parse_args()

    return args
