#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from math import ceil


def prepare_generator_batch(samples, start_letter=0, device="cuda"):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: (batch_size, seq_len). Tensor with a sample in each row

    Returns: inp, target
        - inp: (batch_size, seq_len)
        - target: (batch_size, seq_len). 
    """
    # get the dimensions of the samples tensor
    batch_size, seq_len = samples.size()
    # create a tensor with zeros of dimensions (batch_size, seq_len)
    inp = torch.zeros(batch_size, seq_len)
    # set samples to target
    target = samples
    # set the first column in inp to the start_letter index
    inp[:, 0] = start_letter
    # sets the rest of the tensor to the target
    inp[:, 1:] = target[:, :seq_len-1]
    
    inp = inp.detach().clone()
    target = target.detach().clone() 

    inp = inp.to(device)
    target = target.to(device)

    return inp, target


def prepare_discriminator_data(pos_samples, neg_samples, device="cuda"):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: positive samples of dimensions (num_positive_samples, seq_len)
        - neg_samples: negative samples of dimensions (num_negative_samples, seq_len)

    Returns: inp, target
        - inp: (num_positive_samples + num_negative_samples, seq_len)
        - target: (1, num_positive_samples + num_negative_samples) Either 1 or 0
    """

    num_rows_pos_samples = pos_samples.size()[0]  # get num rows in pos_samples
    num_rows_neg_samples = neg_samples.size()[0]  # get num rows in neg_samples
    # total number of rows in pos_samples and in neg_samples
    num_rows_pos_neg_samples = num_rows_pos_samples + num_rows_neg_samples
    # concatenate pos_samples and neg_samples row-wise
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    # tensor of row of one and has size num_rows_pos_neg_samples
    target = torch.ones(num_rows_pos_neg_samples)
    # convert the non postivie samples to 0
    target[num_rows_pos_samples:] = 0

    # ------- Shuffle ---------
    # random indices of target
    rand_indices = torch.randperm(target.size()[0])
    # shuffle target by passing in the randomized indices
    target = target[rand_indices]
    # order the inp tensor with the new shuffled indices
    inp = inp[rand_indices]
    # remove from graph and clone
    inp = inp.detach().clone()
    # remove from graph and clone
    target = target.detach().clone()
    # change to device
    inp = inp.to(device)
    target = target.to(device)

    return inp, target


def batchwise_sample(gen, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for _ in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len,  device, start_letter=0):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0

    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(
            s[i:i+batch_size], start_letter, device)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)
