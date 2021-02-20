import torch

def Anomaly_score(x, fake_img, z_out_real, D, Lambda=0.1):

    #print('x', x.size())
    #print('fake_img', fake_img.size())
    # テスト画像xと生成画像fake_imgのピクセルレベルの差の絶対値を求めて、ミニバッチごとに和を求める
    residual_loss = torch.abs(x.permute(1, 0, 2)-fake_img.permute(1, 0, 2))
    #print(residual_loss.size())
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    #print(residual_loss.size())
    residual_loss = torch.sum(residual_loss, dim=1)
    #print(residual_loss.size())

    # テスト画像xと生成画像fake_imgを識別器Dに入力し、特徴量マップを取り出す
    #print('x', x.size())
    #print('fake_img', fake_img.size())
    #print('z_out_real', z_out_real.size())
    _, x_feature = D(x, z_out_real, flag_real=True)
    #print('here1')
    _, G_feature = D(fake_img, z_out_real, flag_real=True) #要確認
    #print('here2')

    # テスト画像xと生成画像fake_imgの特徴量の差の絶対値を求めて、ミニバッチごとに和を求める
    discrimination_loss = torch.abs(x_feature-G_feature)
    #print(discrimination_loss.size())
    discrimination_loss = discrimination_loss.view(
        discrimination_loss.size()[0], -1)
    #print(discrimination_loss.size())
    discrimination_loss = torch.sum(discrimination_loss, dim=1)
    #print(discrimination_loss.size())

    # ミニバッチごとに2種類の損失を足し算する
    loss_each = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    #print(loss_each.size())
    # ミニバッチ全部の損失を求める
    total_loss = torch.sum(loss_each)
    #print(total_loss.size())

    return total_loss, loss_each, residual_loss
