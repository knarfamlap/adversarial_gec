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


def train_generator_MLE(gen, gen_opt, oracle, real_data_samples,
                        start_letter, epochs, pos_neg_samples, batch_size, max_seq_len, device):
    """
    MLE Pretraining for Generator
    gen: Generator Object
    gen_opt: Optimizer for the Generator
    real_data_samples: ground truth samples
    """

    for epoch in range(epochs):
        total_loss = 0
        # iterate through the dataset. i has step size batch_size
        for i in range(0, pos_neg_samples, batch_size):
            # prepare the inp and target data
            # inp: (batch_size, max_seq_len)
            # target (batch_size, max_seq_len)
            inp, target = utils.prepare_generator_batch(
                real_data_samples[i:i + batch_size],
                start_letter=start_letter,
                device=device
            )
            # zero the gradients
            gen_opt.zero_grad()
            # get batch loss
            loss = gen.batchNLLLoss(inp, target)
            # backpropagate
            loss.backward()
            # update params
            gen_opt.step()
            # add up the loss
            total_loss += loss.data.item()

            # if (i / batch_size) % ceil(ceil(pos_neg_samples / float(batch_size)) / 10.) == 0:
            #     # LOGGER

        total_loss = total_loss / ceil(
            pos_neg_samples / float(batch_size)) / max_seq_len
        # calculate oraclel loss
        oracle_loss = utils.batchwise_oracle_nll(
            gen,
            oracle,
            pos_neg_samples,
            batch_size,
            max_seq_len,
            start_letter,
            device
        )

        # print the average loss for generator and oracle
        logger.info("Average Train NLL: {}, Oracle Sample NLL: {}".format(
            total_loss, oracle_loss))


def train_generator_PG(gen, gen_opt, oracle, dis, pos_neg_samples, max_seq_len, batch_size, num_batches, device):

    for _ in range(num_batches):
        s = gen.samples(batch_size * 2)

        inp, target = utils.prepare_generator_batch(
            s,
            start_letter,
            device
        )
        # return the reward for the entire sequnce
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    oracle_loss = utils.batchwise_oracle_nll(
        gen,
        oracle,
        pos_neg_samples,
        batch_size,
        max_seq_len,
        start_letter,
        device
    )

    logger.info("Oracle Sample NLL: {}".format(oracle_loss))


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    # get 100 samples
    pos_val = oracle.sample(100)
    # get 100 samples
    neg_val = generator.sample(100)

    val_inp, val_target = utils.prepare_discriminator_data(
        pos_val,
        neg_val,
        device
    )

    for d_step in range(d_steps):
        s = utils.batchwise_sample(generator, pos_neg_samples, batch_size)
        dis_inp, dis_target = utils.prepare_discriminator_data(
            real_data_samples, s, device
        )

        for epoch in range(epochs):
            logger.info('d-step {} epoch {}: '.format(d_step + 1, epoch + 1))
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * pos_neg_samples, batch_size):
                inp, target = dis_inp[i:i +
                                      batch_size], dis_target[i:i + batch_size]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)

                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out > 0.5) ==
                                       (target > 0.5)).data.item()

            total_loss /= ceil(2 * pos_neg_samples / float(batch_size))
            total_acc /= float(2 * pos_neg_samples)

            val_pred = discriminator.batchClassify(val_inp)
            val_acc = torch.sum((val_pred > 0.5) == (
                val_target > 0/5)).data.item() / 200.
            logger.info("Average loss: {}, Train Acc: {}, Val Acc: {}".format(
                total_loss, total_acc, val_acc))


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

    logger.info("Device in user: {}".format(DEVICE))

    oracle = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM,
                       VOCAB_SIZE, MAX_SEQ_LEN, DEVICE)

    oracle = oracle.load_state_dict(torch.load(ORACLE_STATE_DICT_PATH))
    oracle_samples = torch.load(ORACLE_STATE_PATH).type(torch.LongTensor)

    logger.info("Loaded Oracle")

    gen = Generator(
        GEN_EMBEDDING_DIM,
        GEN_HIDDEN_DIM,
        VOCAB_SIZE,
        MAX_SEQ_LEN,
        DEVICE
    )

    logger.info("Loaded Generator")

    dis = Discriminator(
        DIS_EMBEDDING_DIM,
        DIS_HIDDEN_DIM,
        VOCAB_SIZE,
        MAX_SEQ_LEN,
        DEVICE
    )

    logger.info("Loaded Discriminator")

    oracle = oracle.to(DEVICE)
    gen = gen.to(DEVICE)
    dis = dis.to(DEVICE)
    oracle_samples = oracle_samples.to(DEVICE)
    logger.info("Loaded Optimizer for Generator")
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, gen_optimizer, oracle,
                        oracle_samples, MLE_TRAIN_EPOCHS)
    logger.info("Loaded Optimizer for Discriminator")
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, 50, 3)

    oracle_loss = utils.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN, start_letter=START_LETTER, DEVICE)

    for epoch in range(ADV_TRAIN_EPOCHS):
        train_generator_PG(gen, gen_optimizer, oracle, dis, 1)

        train_discriminator(dis, dis_optimizer, oracle_samples, 3)
