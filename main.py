# from math import ceil
# import numpy as np
# import sys
# import pdb

# import torch
# import torch.optim as optim
# import torch.nn as nn

# from generator import Generator
# from discriminator import Discriminator
# import utils


# CUDA = "cuda" if torch.cuda.is_available() else "cpu"
# VOCAB_SIZE = 5000
# MAX_SEQ_LEN = 20
# START_LETTER = 0
# BATCH_SIZE = 32
# MLE_TRAIN_EPOCHS = 100
# ADV_TRAIN_EPOCHS = 50
# POS_NEG_SAMPLES = 10000

# GEN_EMBEDDING_DIM = 32
# GEN_HIDDEN_DIM = 32
# DIS_EMBEDDING_DIM = 64
# DIS_HIDDEN_DIM = 64

# oracle_samples_path = './oracle_samples.trc'
# oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
# pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
# pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'


# def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
#     """
#     MLE Pretraining for Generator
#     gen: Generator Object
#     gen_opt: Optimizer for the Generator
#     real_data_samples: ground truth samples
#     """

#     for epoch in range(epochs):
#         print("Epoch {}: ".format(epoch + 1), end=" ")
#         sys.stdout.flush()
#         total_loss = 0

#         for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
#             inp, target = utils.prepare_generator_batch(
#                 real_data_samples[i:i + BATCH_SIZE],
#                 start_letter=START_LETTER,
#                 gpu=CUDA)
#             gen_opt.zero_grad()
#             loss = gen.batchNLLLoss(inp, target)
#             loss.backward()
#             gen_opt.step()

#             total_loss += loss.data.item()

#             if (i / BATCH_SIZE) % ceil(
#                     ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) /
#                     10.) == 0:  # roughly every 10% of an epoch
#                 print('.', end='')
#                 sys.stdout.flush()

#         total_loss = total_loss / ceil(
#             POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

#         oracle_loss = utils.batchwise_oracle_nll(gen,
#                                                  oracle,
#                                                  POS_NEG_SAMPLES,
#                                                  BATCH_SIZE,
#                                                  MAX_SEQ_LEN,
#                                                  start_letter=START_LETTER,
#                                                  gpu=CUDA)

#         print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' %
#               (total_loss, oracle_loss))


# def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):

#     for _ in range(num_batches):
#         s = gen.samples(BATCH_SIZE * 2)

#         inp, target = utils.prepare_generator_batch(s,
#                                                     start_letter=START_LETTER,
#                                                     gpu=CUDA)
#         rewards = dis.batchClassifiy(target)  # return entire seq reward

#         gen_opt.zero_grad()
#         pg_loss = gen.batchPGLoss(inp, target, rewards)
#         pg_loss.backward()
#         gen_opt.step()

#     oracle_loss = utils.batchwise_oracle_nll(gen,
#                                              oracle,
#                                              POS_NEG_SAMPLES,
#                                              BATCH_SIZE,
#                                              MAX_SEQ_LEN,
#                                              start_letter=START_LETTER,
#                                              gpu=CUDA)

#     print(' oracle_sample_NLL = %.4f' % oracle_loss)


# def train_discriminator(discriminator, dis_opt, real_data_samples, generator,
#                         oracle, d_steps, epochs):
#     pos_val = oracle.sample(100)
#     neg_val = generator.sample(100)

#     val_inp, val_target = utils.prepare_discriminator_data(pos_val,
#                                                            neg_val,
#                                                            gpu=CUDA)

#     for d_step in range(d_steps):
#         s = utils.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
#         dis_inp, dis_target = utils.prepare_discriminator_data(
#             real_data_samples, s, gpu=CUDA)

#         for epoch in range(epochs):
#             print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
#             sys.stdout.flush()
#             total_loss = 0
#             total_acc = 0

#             for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
#                 inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i +
#                                                                     BATCH_SIZE]
#                 dis_opt.zero_grad()
#                 out = discriminator.batchClassify(inp)

#                 loss_fn = nn.BCELoss()
#                 loss = loss_fn(out, target)
#                 loss.backward()
#                 dis_opt.step()

#                 total_loss += loss.data.item()
#                 total_acc += torch.sum(
#                     (out > 0.5) == (target > 0.5)).data.item()

#                 if (i / BATCH_SIZE) % ceil(
#                         ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE)) /
#                         10.) == 0:  # roughly every 10% of an epoch
#                     print('.', end='')
#                     sys.stdout.flush()

#             total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
#             total_acc /= float(2 * POS_NEG_SAMPLES)

#             val_pred = discriminator.batchClassify(val_inp)
#             print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' %
#                   (total_loss, total_acc,
#                    torch.sum(
#                        (val_pred > 0.5) == (val_target > 0.5)).data.item() /
#                    200.))


# if __name__ == '__main__':
#     oracle = Generator(GEN_EMBEDDING_DIM,
#                        GEN_HIDDEN_DIM,
#                        VOCAB_SIZE,
#                        MAX_SEQ_LEN,
#                        gpu=CUDA)
#     oracle.load_state_dict(torch.load(oracle_state_dict_path))
#     oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
#     # a new oracle can be generated by passing oracle_init=True in the generator constructor
#     # samples for the new oracle can be generated using helpers.batchwise_sample()

#     gen = Generator(GEN_EMBEDDING_DIM,
#                     GEN_HIDDEN_DIM,
#                     VOCAB_SIZE,
#                     MAX_SEQ_LEN,
#                     gpu=CUDA)
#     dis = Discriminator(DIS_EMBEDDING_DIM,
#                         DIS_HIDDEN_DIM,
#                         VOCAB_SIZE,
#                         MAX_SEQ_LEN,
#                         gpu=CUDA)

#     if CUDA:
#         oracle = oracle.cuda()
#         gen = gen.cuda()
#         dis = dis.cuda()
#         oracle_samples = oracle_samples.cuda()

#     # GENERATOR MLE TRAINING
#     print('Starting Generator MLE Training...')
#     gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
#     train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples,
#                         MLE_TRAIN_EPOCHS)

#     # torch.save(gen.state_dict(), pretrained_gen_path)
#     # gen.load_state_dict(torch.load(pretrained_gen_path))

#     # PRETRAIN DISCRIMINATOR
#     print('\nStarting Discriminator Training...')
#     dis_optimizer = optim.Adagrad(dis.parameters())
#     train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 50, 3)

#     # torch.save(dis.state_dict(), pretrained_dis_path)
#     # dis.load_state_dict(torch.load(pretrained_dis_path))

#     # ADVERSARIAL TRAINING
#     print('\nStarting Adversarial Training...')
#     oracle_loss = utils.batchwise_oracle_nll(gen,
#                                              oracle,
#                                              POS_NEG_SAMPLES,
#                                              BATCH_SIZE,
#                                              MAX_SEQ_LEN,
#                                              start_letter=START_LETTER,
#                                              gpu=CUDA)
#     print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

#     for epoch in range(ADV_TRAIN_EPOCHS):
#         print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
#         # TRAIN GENERATOR
#         print('\nAdversarial Training Generator : ', end='')
#         sys.stdout.flush()
#         train_generator_PG(gen, gen_optimizer, oracle, dis, 1)

#         # TRAIN DISCRIMINATOR
#         print('\nAdversarial Training Discriminator : ')
#         train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 5,
#                             3)
