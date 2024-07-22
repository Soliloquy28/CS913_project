import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from Database import training_dataset_dataloader, validation_dataset_dataloader, testing_dataset_dataloader
from SleepPPGNet import SleepPPGNet
from CNN import CNN
from LSTM import LSTM
from TCN import TCN
from Training import training_part
from Testing import testing_part
from collections import Counter
import matplotlib.pyplot as plt
import gc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# CNN
# model_cnn = CNN().to(device)

# print('CNN Training')
# training_part(
#     model=model_cnn, 
#     learning_rate=0.001, 
#     num_epochs=50,
#     model_name='CNN'
# )

# print('CNN Testing')
# testing_part(
#     model=model_cnn, 
#     learning_rate=0.001, 
#     model_name='CNN'
# )


# SleepPPG-Net
model_sleepppgnet = SleepPPGNet().to(device)

# print('SleepPPG-Net Training')
# training_part(
#     model=model_sleepppgnet, 
#     learning_rate=0.0005, 
#     num_epochs=50,
#     model_name='SleepPPG-Net'
# )

print('SleepPPG-Net Testing')
testing_part(
    model=model_sleepppgnet, 
    learning_rate=0.0005, 
    model_name='SleepPPG-Net'
)


# LSTM
# model_lstm = LSTM().to(device)

# print('LSTM Training')
# training_part(
#     model=model_lstm, 
#     learning_rate=0.01, 
#     num_epochs=50,
#     model_name='LSTM'
# )

# print('LSTM Testing')
# testing_part(
#     model=model_lstm, 
#     learning_rate=0.001, 
#     model_name='LSTM'
# )


# # TCN
# input_size = 128
# output_size = 128
# kernel_size = 7
# dropout = 0.2

# model_tcn = TCN(input_size, output_size, kernel_size, dropout).to(device)

# print('TCN Training')
# training_part(
#     model=model_tcn, 
#     learning_rate=0.001, 
#     num_epochs=50,
#     model_name='TCN'
# )

# print('TCN Testing')
# testing_part(
#     model=model_tcn, 
#     learning_rate=0.001, 
#     model_name='TCN'
# )