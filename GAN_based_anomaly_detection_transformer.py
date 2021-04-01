#!/usr/bin/env python
# coding: utf-8

# パッケージのimport
import random
import math
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../')
import make_dataset

from make_dataset import _col_names, GAN_Img_Dataset, get_train, get_valid, get_test
from anomaly_score_transformer import Anomaly_score

# Setup seeds
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#PositionalEncoderの実装


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(seq_len, d_model)
        #print('pe', pe.shape)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        #print('position', position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #print('div_term', div_term.shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# # Generatorの実装

class Generator(nn.Module):

    def __init__(self, d_model, seq_len, nhead, dim_feedforward, dropout, num_layers):
        super(Generator, self).__init__()

        self.inp = torch.nn.Parameter(torch.randn(seq_len, 51))
        self.inp.requires_grad = True
        
        # memoryの処理(実データを使わないときに使用)
        self.z_layer = nn.Linear(d_model, d_model*seq_len)
        self.z_norm = nn.LayerNorm(d_model) # Transformer-Encoderの処理に合わせる       
 
        # 時系列データの入力処理
        self.embedding_layer = nn.Linear(51, d_model)
       
        # Positional Encodeingの処理
        self.positionalencoding_layer = PositionalEncoding(d_model, seq_len) 
        
        # Dropoutの処理
        self.dropout_layer = nn.Dropout(p=dropout) 

        # Transformer decoder側の処理 default dim_feedforward =2048
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
    
        #時系列データの出力処理
        self.output_layer = nn.Linear(d_model, 51)   

    def forward(self, z, flag_real):

        #print('Gen')
        #print('z', z.size())
        # 実データを使わない場合
        if flag_real == False:
            memory = self.z_layer(z)
            memory = memory.view(z.size()[0], seq_len, d_model).permute(1, 0, 2)
            memory = self.z_norm(memory)
        # 実データを使う場合
        else:
            memory = z
        #print('memory', memory.size())

        # Decoder側の入力<START>：学習すべきパラメータ
        mini_batch_size = memory.size()[1]
        #print('mini_batch_size', mini_batch_size) 
        inp = torch.cat([self.inp.unsqueeze(0) for _ in range(mini_batch_size)], dim=0)
        inp = inp.permute(1, 0, 2)
        #print('inp', inp.size())           

        x2 = self.embedding_layer(inp)
        x3 = self.positionalencoding_layer(x2)
        #print('x3', x3.size())
        #out = self.decoder_layer(x3, memory) 
        out = self.transformer_decoder(x3, memory)
        #print('out', out.size())
        out = self.output_layer(out)
        #print('out', out.size())

        return out

# # Discriminatorの実装

class Discriminator(nn.Module):

    def __init__(self, x_dim, d_model):
        super(Discriminator, self).__init__()

        # 時系列データ側の入力処理
        self.x_layer = nn.GRU(x_dim, 100)

        # 潜在変数側の入力処理 
        self.z_layer = nn.Sequential(
            nn.Linear(d_model, 100),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.2))        
 
        # 最後の判定
        self.last1 = nn.Sequential( 
            nn.Linear(200, 100),
            nn.LeakyReLU(0.1, inplace=True))
        self.dropout = nn.Dropout(p=0.2)

        self.last2 = nn.Linear(100, 1)

    def forward(self, x, z):

        #print('Dis')
        # 時系列データ側の処理
        _, x_out = self.x_layer(x)
        x_out = x_out.view(x_out.size()[1], x_out.size()[2])
        #print('x', x.size())
        #print('x_out', x_out.size())

        # 潜在変数側の処理
        #print('z', z.size())
        z = self.z_layer(z)
        #print('z', z.size()) 

        # z_outとx_outを結合し、全結合層で判定
        out = torch.cat([x_out, z], dim=1)
        #print('out_x+z', out.size())
        out = self.last1(out)   
        #print('out', out.size())

        feature = out  # 最後にチャネルを1つに集約する手前の情報
        #print(feature.size())

        out = self.dropout(out)

        out = self.last2(out)
        #print('out', out.size())

        return out, feature


# # Encoderの実装

class Encoder(nn.Module):

    def __init__(self, d_model, seq_len, nhead, dim_feedforward, dropout, num_layers):
        super(Encoder, self).__init__()

        # 時系列データの入力処理
        self.embedding_layer = nn.Linear(51, d_model)

        # Positional Encodeingの処理
        self.positionalencoding_layer = PositionalEncoding(d_model, seq_len)
        
        # Dropoutの処理
        self.dropout_layer = nn.Dropout(p=dropout)
        
        # Transformer decoder側の処理 default dim_feedforward =2048
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        # Discriminatorへの入力用
        self.linear_layer_dis = nn.Linear(seq_len*d_model, d_model) 

    def forward(self, x):
              
        #print('Enc')
        #print('inp', x.size())
        x2 = self.embedding_layer(x)
        x3 = self.positionalencoding_layer(x2)
        x4 = self.dropout_layer(x3)
        #out = self.encoder_layer(x4)
        out = self.transformer_encoder(x4)
        #print('out', out.size())

        # Discriminatorへの入力用
        mini_batch_size = out.size()[1]
        out2 = out.view(mini_batch_size, -1)
        #print('out2', out2.size())
        out2 = self.linear_layer_dis(out2)
        #print('out2', out2.size())        

        return out, out2


# モデルを学習させる関数を作成

def train_model(G, D, E, train_dataloader, valid_dataloader, num_epochs, d_model, file_dir):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("使用デバイス：", device)

    # 最適化手法の設定
    lr_ge = 0.0001
    lr_d = 0.0001 / 4
    print('learning rate of generator and encoder: {}'.format(lr_ge))
    print('learning rate of discriminator: {}'.format(lr_d))
    beta1, beta2 = 0.5, 0.999
    g_optimizer = torch.optim.Adam(G.parameters(), lr_ge, [beta1, beta2])
    e_optimizer = torch.optim.Adam(E.parameters(), lr_ge, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])

    # 誤差関数を定義
    # BCEWithLogitsLossは入力にシグモイド（logit）をかけてから、
    # バイナリークロスエントロピーを計算
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # パラメータをハードコーディング
    z_dim = 100

    # ネットワークをGPUへ
    G.to(device)
    E.to(device)
    D.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    best_f1 = 0
    best_epoch = 0

    # epochのループ
    for epoch in range(num_epochs):

        G.train()  # モデルを訓練モードに
        E.train()  # モデルを訓練モードに
        D.train()  # モデルを訓練モードに

        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_loss = 0.0  # epochの損失和
        epoch_e_loss = 0.0  # epochの損失和
        epoch_d_loss = 0.0  # epochmini_batch_size = timeseries.size()[0]の損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        # データローダーからminibatchずつ取り出すループ
        for timeseries, labels in train_dataloader:
            #print(timeseries.size())
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if timeseries.size()[0] == 1:
                continue

            # ミニバッチサイズの1もしくは0のラベル役のテンソルを作成
            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチの数が少なくなる
            mini_batch_size = timeseries.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            # GPUが使えるならGPUにデータを送る
           # timeseris = timeseries.to(device)

            # --------------------
            # 1. Discriminatorの学習
            # --------------------
            # 真の画像を判定
            timeseries = timeseries.permute(1, 0, 2).to(device)
            #print(timeseries.size())
            _, z_out_real = E(timeseries)
            #print(z_out_real.size())
            d_out_real, _ = D(timeseries, z_out_real)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, d_model).to(device)       
            fake_images = G(input_z, flag_real=False)
            #print(fake_images.size())
            d_out_fake, _ = D(fake_images, input_z)

            # 誤差を計算
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake
            #print(d_loss.size())

            # バックプロパゲーション
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # --------------------
            # 2. Generatorの学習
            # --------------------
            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, d_model).to(device)
            fake_images = G(input_z, flag_real=False)
            d_out_fake, _ = D(fake_images, input_z)

            # 誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # バックプロパゲーション
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. Encoderの学習
            # --------------------
            # 真の画像のzを推定
            _, z_out_real = E(timeseries)
            #print(z_out_real.size())
            d_out_real, _ = D(timeseries, z_out_real)

            # 誤差を計算
            e_loss = criterion(d_out_real.view(-1), label_fake)

            # バックプロパゲーション
            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            # --------------------
            # 4. 記録
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_e_loss += e_loss.item()
            iteration += 1

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f} ||Epoch_E_Loss:{:.4f}'.format(
            epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size, epoch_e_loss/batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()


        # validation

        G.eval()  # モデルを検証モードに
        E.eval()  # モデルを検証モードに
        scores_valid = [] # Anomaly Score of valid dataset
        labels_valid = []

        # データローダーからminibatchずつ取り出すループ
        for timeseries, labels in valid_dataloader:
            #print(timeseries.size())
            #print(labels.size())
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if timeseries.size()[0] == 1:
                continue

            # epochの最後のイテレーションはミニバッチの数が少なくなる
            mini_batch_size = timeseries.size()[0]
            #print(mini_batch_size)
            # 異常検知したいデータをエンコードしてzにしてから、Gで生成
            timeseries = timeseries.permute(1, 0, 2).to(device)
            z_out_real, z_out_real_2 = E(timeseries)
            #print(z_out_real.size())
            imges_reconstract = G(z_out_real, flag_real=True)
            #print('timeseries', timeseries.size())
            #print('imges_reconstract', imges_reconstract.size())

            # 損失を求める
            loss, loss_each, residual_loss_each = Anomaly_score(
                timeseries, imges_reconstract, z_out_real_2, D, Lambda=0.1)

            # 損失の計算。トータルの損失
            loss_each = loss_each.cpu().detach().numpy()
            scores_valid += loss_each.tolist()[:mini_batch_size]
            labels_valid += labels.tolist()[:mini_batch_size]

        # 異常なデータの決定
        per = np.percentile(np.array(scores_valid), 80)
        y_pred = scores_valid.copy()
        y_pred = np.array(y_pred)
        inds = (y_pred < per)
        inds_comp = (y_pred >= per)
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1,_ = precision_recall_fscore_support(np.array(labels_valid),
                                                                  y_pred,
                                                                  average='binary')
        accuracy = accuracy_score(np.array(labels_valid), y_pred)

        print("Validating : Prec = %.4f | Rec = %.4f | F1 = %.4f | Accuracy = %.4f"
                % (precision, recall, f1, accuracy))        

        # PyTorchのネットワークのパラメータの保存
        if epoch == 0 or f1 >= best_f1:
            torch.save(G.state_dict(), '{}G_weights_tf.pth'.format(file_dir))
            torch.save(D.state_dict(), '{}D_weights_tf.pth'.format(file_dir))
            torch.save(E.state_dict(), '{}E_weights_tf.pth'.format(file_dir))
            best_f1 = f1
            best_epoch = epoch
        print('best epoch: ', best_epoch)
        print('best f1: ', best_f1)
        print('--------------------------')
        

    print("総イテレーション回数:", iteration)

    #return G, D, E


# ネットワークの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        # 全結合層Linearの初期化
        m.bias.data.fill_(0)


# 事前設定
d_model = 512 # default=512
seq_len = 24 # default=12
nhead = 8 # default=8
dim_feedforward  = 2048 # default=2048
dropout = 0.1 # default=0.1
num_layers = 6 # default=6
batch_size = 64

print('d_model: {}'.format(d_model))
print('seq_len: {}'.format(seq_len))
print('nhead: {}'.format(nhead))
print('dim_feedforward: {}'.format(dim_feedforward))
print('dropout: {}'.format(dropout))
print('num_layers: {}'.format(num_layers))
print('batch_size: {}'.format(batch_size))

# Datasetを生成
train_dataset = get_train('/Storage/maru/SWaT/shuffle_train_{0}_0.2_{0}.csv'.format(seq_len), seq_len)
valid_dataset = get_valid('/Storage/maru/SWaT/shuffle_valid_{0}_0.2_{0}.csv'.format(seq_len), seq_len)
test_dataset = get_test('/Storage/maru/SWaT/shuffle_test_{0}_0.2_{0}.csv'.format(seq_len), seq_len)
'''
train_dataset = get_train('/home/maru/data/SWaT/shuffle_train_window{0}_shift{0}_0.2_{0}.csv'.format(seq_len), seq_len)
valid_dataset = get_valid('/home/maru/data/SWaT/shuffle_valid_window{0}_shift{0}_0.2_{0}.csv'.format(seq_len), seq_len)
test_dataset = get_test('/home/maru/data/SWaT/shuffle_test_window{0}_shift{0}_0.2_{0}.csv'.format(seq_len), seq_len)
'''

# DataLoaderを作成
train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False)
valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)

# 動作の確認
#batch_iterator = iter(train_dataloader)  # イテレータに変換
#timeseries, labels = next(batch_iterator)  # 1番目の要素を取り出す
#print(timeseries.size())  # torch.Size([batch_size, seq_len, dim])
#print(labels.size()) # torch.Size([batch_size])

#batch_iterator = iter(valid_dataloader)  # イテレータに変換
#timeseries, labels = next(batch_iterator)  # 1番目の要素を取り出す
#print(timeseries)
#print(labels)

G = Generator(d_model, seq_len, nhead, dim_feedforward, dropout, num_layers)
E = Encoder(d_model, seq_len, nhead, dim_feedforward, dropout, num_layers)
D = Discriminator(x_dim=51, d_model=d_model)

# 初期化の実施
G.apply(weights_init)
E.apply(weights_init)
D.apply(weights_init)

print("ネットワークの初期化完了")


today = datetime.now().strftime("%Y%m%d")
file_dir = './weights/v1/{0}/{1}/'.format(seq_len, today)
if not os.path.exists(file_dir):
    os.makedirs(file_dir)


# 学習・検証を実行する
num_epochs = 2
train_model(
G, D, E, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, num_epochs=num_epochs, d_model=d_model, file_dir=file_dir)


# DataLoaderを作成
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


#def test_model(G_update, D_update, E_update, test_dataloader):
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# PyTorchのネットワークパラメータのロード
G_load_weights = torch.load('{}G_weights_tf.pth'.format(file_dir))
D_load_weights = torch.load('{}D_weights_tf.pth'.format(file_dir))
E_load_weights = torch.load('{}E_weights_tf.pth'.format(file_dir))
G.load_state_dict(G_load_weights)
D.load_state_dict(D_load_weights)
E.load_state_dict(E_load_weights)

print('ネットワーク設定完了：学習済みの重みをロードしました')

# テスト
E.eval()
G.eval()
scores_test = [] # Anomaly Score of test dataset
labels_test = []

# データローダーからminibatchずつ取り出すループ
for timeseries, labels in test_dataloader:
    
    # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
    if timeseries.size()[0] == 1:
        continue

    # 異常検知したいデータをエンコードしてzにしてから、Gで生成
    mini_batch_size = timeseries.size()[0]
    timeseries = timeseries.permute(1, 0, 2).to(device)
    z_out_real, z_out_real_2 = E(timeseries)
    #z_out_real = z_out_real.permute(1, 0, 2)
    #z_out_real = z_out_real.reshape(-1, 300).to(device)
    #print('z_out_real', z_out_real.size())
    #print('inp_x', input_x.size())
    imges_reconstract = G(z_out_real, flag_real=True)
    #print(timeseries.size())
    #print(imges_reconstract.size())

    # 損失を求める
    loss, loss_each, residual_loss_each = Anomaly_score(
        timeseries, imges_reconstract, z_out_real_2, D, Lambda=0.1)

    # 損失の計算。トータルの損失
    loss_each = loss_each.cpu().detach().numpy()
    #print("total loss：", np.round(loss_each, 0))
    scores_test += loss_each.tolist()[:mini_batch_size]
    #print(scores_test)
    labels_test += labels.tolist()[:mini_batch_size]

#print(scores_test)
#print(len(scores_test))

# 異常なデータの決定
per = np.percentile(np.array(scores_test), 80)
y_pred = scores_test.copy()
y_pred = np.array(y_pred) 
inds = (y_pred < per)
inds_comp = (y_pred >= per)
y_pred[inds] = 0
y_pred[inds_comp] = 1

#print(len(labels_test))
#print(len(y_pred))
precision, recall, f1,_ = precision_recall_fscore_support(np.array(labels_test).round(),
                                                          y_pred.round(),
                                                          average='binary')
accuracy = accuracy_score(np.array(labels_test).round(), y_pred.round())

print("Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f | Accuracy = %.4f"
        % (precision, recall, f1, accuracy))

