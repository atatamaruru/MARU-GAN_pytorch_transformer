#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np

def _col_names():
    return ['Timestamp','FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206','DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302','AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503','FIT601','P601','P602','P603','Is_Attack']


class GAN_Img_Dataset(data.Dataset):
    """時系列データのDatasetクラス。PyTorchのDatasetクラスを継承"""

    def __init__(self, dataset, labels, seq_len):
        self.dataset = dataset
        self.labels = labels
        self.seq_len = seq_len
        time_series= torch.Tensor()
        self.time_series = time_series.new_zeros((seq_len, dataset.shape[1]))

    def __len__(self):
        '''時系列データの数を返す'''
        return int(len(self.dataset) / self.seq_len)

    def __getitem__(self, index):
        '''前処理をした画像のTensor形式のデータを取得'''
        return self.dataset[index*self.seq_len:(index+1)*self.seq_len], self.labels[index*self.seq_len:(index+1)*self.seq_len].max()
        #for i in range(self.seq_len):
        #    self.time_series[i] = self.dataset[(index*self.seq_len)+i]
        #label =self.labels[index*self.seq_len:(index*self.seq_len)+self.seq_len].max()       
       
        #return self.time_series, label


# Datasetを作成

#時系列データを生成

def get_train(file_path, seq_len):
    
    train_df_dataset = pd.read_csv(file_path, delimiter=',', names=_col_names())
    train_df_dataset = train_df_dataset.replace({'Is_Attack': {'Normal':0}})
    train_nmp_dataset, train_nmp_labels = np.array(train_df_dataset.iloc[1:,1:52].astype(np.float32)), np.array(train_df_dataset.iloc[1:,52:53].astype(np.float32))
    train_dataset = GAN_Img_Dataset(torch.tensor(train_nmp_dataset), torch.tensor(train_nmp_labels), seq_len)
    return train_dataset


def get_valid(file_path, seq_len):

    #valid_file_path = '/home/maru/data/SWaT/shuffle_valid_12_0.2_12.csv'
    valid_df_dataset = pd.read_csv(file_path, delimiter=',', names=_col_names())
    valid_nmp_dataset, valid_nmp_labels = np.array(valid_df_dataset.iloc[1:,1:52].astype(np.float32)), np.array(valid_df_dataset.iloc[1:,52:53].astype(np.float32))
    valid_dataset = GAN_Img_Dataset(torch.tensor(valid_nmp_dataset), torch.tensor(valid_nmp_labels), seq_len)
    #print(valid_nmp_dataset.shape)
    #print(valid_nmp_labels.shape)
    return valid_dataset


def get_test(file_path, seq_len):

    #test_file_path = '/home/maru/data/SWaT/shuffle_test_12_0.2_12.csv'
    test_df_dataset = pd.read_csv(file_path, delimiter=',', names=_col_names())
    test_nmp_dataset, test_nmp_labels = np.array(test_df_dataset.iloc[1:,1:52].astype(np.float32)), np.array(test_df_dataset.iloc[1:,52:53].astype(np.float32))
    test_dataset = GAN_Img_Dataset(torch.tensor(test_nmp_dataset), torch.tensor(test_nmp_labels), seq_len)

    return test_dataset

'''
# 動作確認
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

# 動作の確認
batch_iterator = iter(train_dataloader)  # イテレータに変換
inputs, labels = next(batch_iterator)  # 1番目の要素を取り出す
print(inputs.size())
print(labels.size())
'''
