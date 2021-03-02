from math import ceil
import numpy as np
import sys
import pdb
from logzero import logger
import argparse

import torch
import torch.optim as optim
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sequence GAN")

    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--start_letter", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--mle_train_epochs", type=int)
    parser.add_argument("--adv_train_epochs", type=int)
    parser.add_argument("--pos_neg_samples", type=int)
    parser.add_argument("--gen_embedding_dim", type=int)
    parser.add_argument("--gen_hidden_dim", type=int)
    parser.add_argument("--dis_embedding_dim", type=int)
    parser.add_argument("--dis_hidden_dim", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--oracle_state_path", type=str)
    parser.add_argument("--oracle_state_dict_path", type=str)
    parser.add_argument("--pretrained_gen_path", type=str)
    parser.add_argument("--pretrained_dis_path", type=str)

    args = parser.parse_args()

    VOCAB_SIZE = args.vocab_size
    MAX_SEQ_LEN = args.max_seq_len
    START_LETTER = args.start_letter
    BATCH_SIZE = args.batch_size
    MLE_TRAIN_EPOCHS = args.mle_train_epochs
    ADV_TRAIN_EPOCHS = args.adv_train_epochs
    POS_NEG_SAMPLES = args.pos_neg_samples

    GEN_EMBEDDING_DIM = args.gen_embedding_dim
    GEN_HIDDEN_DIM = args.gen_hidden_dim
    DIS_EMBEDDING_DIM = args.dis_embedding_dim
    DIS_HIDDEN_DIM = args.dis_hidden_dim

    ORACLE_STATE_PATH = args.oracle_state_path
    ORACLE_STATE_DICT_PATH = args.oracle_state_dict_path
    PRETRAINED_GEN_PATH = args.pretrained_gen_path
    PRETRAINED_DIS_PATH = args.pretrained_dis_path

    if args.device == "cpu":
        DEVICE = "cpu"
    else:
        DEVICE = "cuda:{}".format(
            args.device) if torch.cuda.is_available() else "cpu"

    oracle = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM,
                       VOCAB_SIZE, MAX_SEQ_LEN)

    oracle = oracle.load_state_dict(torch.load(ORACLE_STATE_DICT_PATH))
    oracle_samples = torch.load(ORACLE_STATE_PATH).type(torch.LongTensor)

    gen = Generator(
        GEN_EMBEDDING_DIM,
        GEN_HIDDEN_DIM,
        VOCAB_SIZE,
        MAX_SEQ_LEN
    )

    dis = Discriminator(
        DIS_EMBEDDING_DIM,
        DIS_HIDDEN_DIM,
        VOCAB_SIZE,
        MAX_SEQ_LEN
    )

    oracle = oracle.to(DEVICE)
    gen = gen.to(DEVICE)
    dis = dis.to(DEVICE)
    oracle_samples = oracle_samples.to(DEVICE)

    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, gen_optimizer, oracle,
                        oracle_samples, MLE_TRAIN_EPOCHS)
