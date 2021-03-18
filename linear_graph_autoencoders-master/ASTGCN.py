# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs
from time import time
import os
import numpy as np
import argparse
import configparser
from torch.utils.tensorboard import SummaryWriter
import csv

#解决预测值存在0等非法值的MSE计算方式
def masked_mape_np(y_true, y_pred, null_val=np.nan):
    #防止分母为0导致nan报警告
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

#逆Normalization
def re_normalization(x, mean, std):
    x = x * std + mean
    return x

#获得道路拓扑结构的邻接矩阵
def get_adjacency_matrix(distance_df_filename, num_of_loads, id_filename=None):
    '''
    distance_df_filename:（str）文件名
    num_of_vertices:（int）道路的数量
    返回：A代表路之间累计经过多少车
    distaneA代表车从A到B的开销
    '''

    #如果数据是npy文件则直接读取
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    #如果文件时excel文件
    else:
        A = np.zeros((int(num_of_loads), int(num_of_loads)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_loads), int(num_of_loads)), dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

#用于切比雪夫多项式近似中的拉普拉斯计算,eigs函数能返回矩阵的特征值/特征向量
def scaled_Laplacian(W):
    '''
    W: （N,N）矩阵
    返回：拉普拉斯变化后的（N,N）矩阵
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

#计算切比雪夫多项式
def cheb_polynomial(L_tilde, K):
    '''
    L_tilde: 比例拉普拉斯算子(N, N)矩阵
    K: 切比雪夫多项式最大项数
    返回切比雪夫多项式：from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

#导入PEMS数据集，返回训练集、验证集、测试集
def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    从文件读入的数据，x是最大最小归一化的，但是y是真实值
    返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)
    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)
    filename = os.path.join(dirpath,file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w'
                            + str(num_of_weeks)) +'_astcgn'
    print('load file:', filename)
    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std

#计算模型的验证损失
def compute_val_loss_astgcn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_stepa
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  #保证Dropout层不在训练模式
    with torch.no_grad():
        val_loader_length = len(val_loader)  # nb of batch
        tmp = []  # 记录了所有batch的loss
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss

#计算测试集RMSE、RMSE、MAPE
def predict_and_save_results_astgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    '''
    net.train(False)
    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)  # nb of batch
        prediction = []  # 存储所有batch的output
        input = []  # 存储所有batch的input
        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, labels = batch_data
            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)
            outputs = net(encoder_inputs)
            prediction.append(outputs.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)
        input = re_normalization(input, _mean, _std)
        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

#空间注意力层，计算空间注意力分数
class Spatial_Attention_layer(nn.Module):

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized

#K-切比雪夫图卷积带注意力机制
class cheb_conv_withSAt(nn.Module):

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        batch_size, num_of_loads, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_loads, self.out_channels).to(self.DEVICE)  # (b, N, F_out)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

#时间注意力机制层
class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))

#模块搭建（TAt -> SAt -> GCN -> Conv)+残差模块
class ASTGCN_block(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  #需要将channel放到最后一个维度上

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)# (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class ASTGCN_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        super(ASTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([ASTGCN_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, len_input)])
        self.BlockList.extend([ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, num_of_vertices, len_input//time_strides) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = ASTGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model



if __name__ == "__main__":

    #读取配置文件设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/PEMS08_astgcn.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']


    #数据读取参数
    adj_filename = data_config['adj_filename']
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    if config.has_option('Data', 'id_filename'):
        id_filename = data_config['id_filename']
    else:
        id_filename = None
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    dataset_name = data_config['dataset_name']


    #训练参数设置
    model_name = training_config['model_name']
    ctx = training_config['ctx']
    os.environ["CUDA_VISIBLE_DEVICES"] = ctx
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0')
    print("CUDA:", USE_CUDA, DEVICE)
    learning_rate = float(training_config['learning_rate'])
    epochs = int(training_config['epochs'])
    start_epoch = int(training_config['start_epoch'])
    batch_size = int(training_config['batch_size'])
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    time_strides = num_of_hours
    nb_chev_filter = int(training_config['nb_chev_filter'])
    nb_time_filter = int(training_config['nb_time_filter'])
    in_channels = int(training_config['in_channels'])
    nb_block = int(training_config['nb_block'])
    K = int(training_config['K'])

    #创建保存参数目录
    folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
    print('folder_dir:', folder_dir)
    params_path = os.path.join('experiments', dataset_name, folder_dir)
    print('params_path:', params_path)

    #构建数据集和模型
    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
        graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
    net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices)

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(flush_secs=5)
    # summary(net, (170, 1, 12))

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    # train model
    # for epoch in range(0, epochs):
    #     params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
    #     val_loss = compute_val_loss_astgcn(net, val_loader, criterion, sw, epoch)
    #
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_epoch = epoch
    #         torch.save(net.state_dict(), params_filename)
    #         print('save parameters to file: %s' % params_filename)
    #
    #     net.train()
    #     for batch_index, batch_data in enumerate(train_loader):
    #         encoder_inputs, labels = batch_data
    #         optimizer.zero_grad()
    #         outputs = net(encoder_inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         training_loss = loss.item()
    #         global_step += 1
    #         sw.add_scalar('training_loss', training_loss, global_step)
    #         if global_step % 1000 == 0:
    #             print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    #测试过程
    a = 14
    params_filename = os.path.join(params_path, 'epoch_%s.params' % a)
    print('load weight from:', params_filename)
    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results_astgcn(net, test_loader
                                    , test_target_tensor, global_step, _mean, _std, params_path, 'test')