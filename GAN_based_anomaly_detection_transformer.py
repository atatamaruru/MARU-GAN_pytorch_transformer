#!/usr/bin/env python
# coding: utf-8

# パッケージのimport
import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from make_dataset import _col_names, GAN_Img_Dataset, get_valid, get_test
from anomaly_score_transformer import Anomaly_score

# Setup seeds
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#PositionalEncoderの実装

'''
class PositionalEncoder(nn.Module):
    #入力された単語の位置を示すベクトル情報を付加する

    def __init__(self, d_model=512, max_seq_len=12):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る、ここでは省略。実際に学習時には使用する
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))

                # 誤植修正_200510 #79
                # pe[pos, i + 1] = math.cos(pos /
                #                          (10000 ** ((2 * (i + 1))/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        #print(x.size())
        #print(self.pe.size())
        #print(x.device)
        #print(self.pe.device)
        ret = math.sqrt(self.d_model)*x + self.pe.to(device='cuda:1')
        
        return ret
'''

class PositionalEncoding(nn.Module):
 
    def __init__(self, d_model, seq_len=12):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
 
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #print('pe', pe.size())
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# # Generatorの実装

class Generator(nn.Module):

    def __init__(self, d_model):
        super(Generator, self).__init__()

        self.inp = torch.nn.Parameter(torch.randn(12, 51))
        self.inp.requires_grad = True
        
        # memoryの処理(実データを使わないときに使用)
        self.z_layer = nn.Linear(512, 512*12)       
 
        # 時系列データの入力処理
        self.embedding_layer = nn.Linear(51, d_model)
       
        # Positional Encodeingの処理
        self.positionalencoding_layer = PositionalEncoding(d_model, seq_len=12) 
        
        # Transformer decoder側の処理 default dim_feedforward =2048
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
     
        #時系列データの出力処理
        self.output_layer = nn.Linear(512, 51)   

    def forward(self, z, flag_real):

        #print('Gen')
        #print('z', z.size())
        # 実データを使わない場合
        if flag_real == False:
            memory = self.z_layer(z).view(12, z.size()[0], 512)
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
        out = self.decoder_layer(x3, memory) 
        #print('out', out.size())
        out = self.output_layer(out)
        #print('out', out.size())

        return out

# # Discriminatorの実装

class Discriminator(nn.Module):

    def __init__(self, x_dim, d_model):
        super(Discriminator, self).__init__()

        # 時系列データ側の入力処理
        self.x_layer = nn.GRU(x_dim, 100, dropout=0.2)

        # 潜在変数側の入力処理
        # 実データを使う場合
        self.z_layer_real = nn.Linear(12*d_model, 100)
        # 実データを使わない場合    
        self.z_layer_fake = nn.Linear(d_model, 100)        
 
        # 最後の判定
        self.last1 = nn.Sequential( 
            nn.Linear(200, 100),
            nn.LeakyReLU(0.1, inplace=True))
        self.dropout = nn.Dropout(p=0.2)

        self.last2 = nn.Linear(100, 1)

    def forward(self, x, z, flag_real):

        #print('Dis')
        # 時系列データ側の処理
        _, x_out = self.x_layer(x)
        x_out = x_out.view(x_out.size()[1], x_out.size()[2])
        #print('x', x.size())
        #print('x_out', x_out.size())

        # 潜在変数側の処理
        batch_size = z.size()[1]
        if flag_real:
            z = z.permute(1, 0, 2).contiguous().view(batch_size, -1)
            z = self.z_layer_real(z)
        
        if flag_real == False:        
            z = self.z_layer_fake(z)

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

    def __init__(self, d_model):
        super(Encoder, self).__init__()

        # 時系列データの入力処理
        self.embedding_layer = nn.Linear(51, d_model)

        # Positional Encodeingの処理
        self.positionalencoding_layer = PositionalEncoding(d_model, seq_len=12)

        # Transformer decoder側の処理 default dim_feedforward =2048
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)

    def forward(self, x):
              
        #print('Enc')
        #print('inp', x.size())
        x2 = self.embedding_layer(x)
        x3 = self.positionalencoding_layer(x2)
        out = self.encoder_layer(x3)
        #print('out', out.size())        

        return out


# Datasetを作成
seq_len = 12

train_file_path = '/home/maru/data/SWaT/shuffle_train_12_0.2_12.csv'
train_df_dataset = pd.read_csv(train_file_path, delimiter=',', names=_col_names())
train_df_dataset = train_df_dataset.replace({'Is_Attack': {'Normal':0}})
train_nmp_dataset, train_nmp_labels = np.array(train_df_dataset.iloc[1:,1:52].astype(np.float32)), np.array(train_df_dataset.iloc[1:,52:53].astype(np.float32))
train_dataset = GAN_Img_Dataset(torch.tensor(train_nmp_dataset), torch.tensor(train_nmp_labels), seq_len)

valid_dataset = get_valid(seq_len)
test_dataset = get_test(seq_len)


# DataLoaderを作成
batch_size = 64

train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
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


# モデルを学習させる関数を作成

def train_model(G, D, E, train_dataloader, valid_dataloader, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #print("使用デバイス：", device)

    # 最適化手法の設定
    lr_ge = 0.0001
    lr_d = 0.0001/4
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
    seq_len = 12

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
            z_out_real = E(timeseries)
            #print(z_out_real.size())
            d_out_real, _ = D(timeseries, z_out_real, flag_real=True)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, 512).to(device)       
            fake_images = G(input_z, flag_real=False)
            #print(fake_images.size())
            d_out_fake, _ = D(fake_images, input_z, flag_real=False)

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
            input_z = torch.randn(mini_batch_size, 512).to(device)
            fake_images = G(input_z, flag_real=False)
            d_out_fake, _ = D(fake_images, input_z, flag_real=False)

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
            z_out_real = E(timeseries)
            #print(z_out_real.size())
            d_out_real, _ = D(timeseries, z_out_real, flag_real=True)

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
            z_out_real = E(timeseries)
            #print(z_out_real.size())
            imges_reconstract = G(z_out_real, flag_real=True)
            #print('timeseries', timeseries.size())
            #print('imges_reconstract', imges_reconstract.size())

            # 損失を求める
            loss, loss_each, residual_loss_each = Anomaly_score(
                timeseries, imges_reconstract, z_out_real, D, Lambda=0.1)

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
            torch.save(G.state_dict(), './weights/transformer/G_weights_tf.pth')
            torch.save(D.state_dict(), './weights/transformer/D_weights_tf.pth')
            torch.save(E.state_dict(), './weights/transformer/E_weights_tf.pth')
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

G = Generator(d_model=512)
E = Encoder(d_model=512)
D = Discriminator(x_dim=51, d_model=512)

# 初期化の実施
G.apply(weights_init)
E.apply(weights_init)
D.apply(weights_init)

print("ネットワークの初期化完了")


# 学習・検証を実行する
num_epochs = 10000
train_model(
G, D, E, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, num_epochs=num_epochs)


# DataLoaderを作成
batch_size = 64

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


#def test_model(G_update, D_update, E_update, test_dataloader):
 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# PyTorchのネットワークパラメータのロード
G_load_weights = torch.load('./weights/transformer/G_weights_tf.pth')
D_load_weights = torch.load('./weights/transformer/D_weights_tf.pth')
E_load_weights = torch.load('./weights/transformer/E_weights_tf.pth')
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
    z_out_real = E(timeseries)
    #z_out_real = z_out_real.permute(1, 0, 2)
    #z_out_real = z_out_real.reshape(-1, 300).to(device)
    #print('z_out_real', z_out_real.size())
    #print('inp_x', input_x.size())
    imges_reconstract = G(z_out_real, flag_real=True)
    #print(timeseries.size())
    #print(imges_reconstract.size())

    # 損失を求める
    loss, loss_each, residual_loss_each = Anomaly_score(
        timeseries, imges_reconstract, z_out_real, D, Lambda=0.1)

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

