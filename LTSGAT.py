import torch.nn as nn
from grl import ReverseLayerF
import numpy as np
from einops import rearrange
from Electrodes import Electrodes
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GATConv,SGConv,TAGConv
import torch
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Optional, Tuple



class Attention_Layer1(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self):
        super(Attention_Layer1, self).__init__()
        # 下面使用nn的Linear层来定义Q，K，V矩阵
        self.Q_linear = nn.Linear(32, 32, bias=False)
        self.K_linear = nn.Linear(32, 32, bias=False)
        self.V_linear = nn.Linear(32, 32, bias=False)

    def forward(self, data):
        # 计算生成QKV矩阵
        att_input = data.x
        att_input = rearrange(att_input, '(b i) sl -> b sl i', i=32)
        Q = self.Q_linear(att_input)
        K = self.K_linear(att_input).permute(0,2,1)  # 先进行一次转置
        V = self.V_linear(att_input)

        # 下面开始计算啦
        alpha = torch.matmul(Q, K)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)
        out = torch.matmul(alpha, V)
        return out

"""
class GradientReverseFunction(Function):

    # 写自定义的梯度计算方式

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None
"""

class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size):#, sequence_length
        super(BiLSTM, self).__init__()
        self.hidden_size= hidden_size

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True, bidirectional=True)


    def forward(self, x):
        # x = data.x
        x = rearrange(x,'b sl i -> b i sl',i=32)
        x, _ = self.lstm(x)#x (b,32,128)
        return x


def Region_apart(x):
    region1 = torch.cat((x[:,0,:],x[:,1,:],x[:,16,:],x[:,17,:]),dim=1)
    region2 = torch.cat((x[:, 2, :], x[:, 3, :], x[:, 18, :], x[:, 19, :],x[:, 20, :]), dim=1)
    region3 = torch.cat((x[:,7, :], x[:, 25, :]), dim=1)
    region4 = torch.cat((x[:, 6, :], x[:,23, :], x[:, 24, :]), dim=1)
    region5 = torch.cat((x[:, 4, :], x[:, 5, :], x[:, 21, :], x[:,22, :]), dim=1)
    region6 = torch.cat((x[:, 8, :], x[:,9, :], x[:, 26, :], x[:, 27, :]), dim=1)
    region7 = torch.cat((x[:, 10, :], x[:, 11, :], x[:, 15, :], x[:,28, :],x[:, 29, :]), dim=1)
    region8 = torch.cat((x[:,12, :], x[:,30, :]), dim=1)
    region9 = torch.cat((x[:,13, :], x[:,14, :], x[:,31, :]), dim=1)
    # region1 = region1.reshape(batch, 4, 128)
    # region2 = region2.reshape(batch, 5, 128)
    # region3 = region3.reshape(batch, 2, 128)
    # region4 = region4.reshape(batch, 3, 128)
    # region5 = region5.reshape(batch, 4, 128)
    # region6 = region6.reshape(batch, 4, 128)
    # region7 = region7.reshape(batch, 5, 128)
    # region8 = region8.reshape(batch, 2, 128)
    # region9 = region9.reshape(batch, 3, 128)
    return region1,region2,region3,region4,region5,region6,region7,region8,region9

class Attention_Layer(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self):
        super(Attention_Layer, self).__init__()
        # 下面使用nn的Linear层来定义Q，K，V矩阵
        self.Q_linear = nn.Linear(256//2, 256//2, bias=False)
        self.K_linear = nn.Linear(256//2, 256//2, bias=False)
        self.V_linear = nn.Linear(256//2, 256//2, bias=False)

    def forward(self, att_input):
        # 计算生成QKV矩阵
        Q = self.Q_linear(att_input)
        K = self.K_linear(att_input).permute(0, 2, 1)  # 先进行一次转置
        V = self.V_linear(att_input)

        # 下面开始计算啦
        alpha = torch.matmul(Q, K)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)
        out = torch.matmul(alpha, V)
        return out

class GAT(torch.nn.Module):
    def __init__(self, num_features):
        super(GAT, self).__init__()
        self.gat1 = GATConv(num_features, 16, heads=4,dropout=0.6)
        self.gat2 = GATConv(64, 16, heads=4,dropout=0.6)#
        self.gat3 = GATConv(64, 16, heads=4,dropout=0.6)
        self.gat4 = GATConv(64, 64, heads=1,dropout=0.6)
        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        # self.dropout3 = nn.Dropout(0.5)
        # self.dropout4 = nn.Dropout(0.5)
        # self.fc = nn.Linear(16, 2)

    def forward(self,data):
        x,edge_index,batch = data.x,data.edge_index, data.batch
        x = F.leaky_relu(self.gat1(x, edge_index))

        x = F.leaky_relu(self.gat2(x, edge_index))

        x = F.leaky_relu(self.gat3(x, edge_index))
        #
        x = F.leaky_relu(self.gat4(x, edge_index))

        x = gap(x,batch)
        # x = torch.cat((gmp(x, batch), gap(x, batch)), dim=1)
        # logits = self.fc(x)
        # probas = F.softmax(logits,dim=1)#, dim=1
        return x

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.attention1 = Attention_Layer1()
        self.LSTM = BiLSTM(10,64//2)
        self.dropout1 = nn.Dropout(0.5)


        self.linear1 = nn.Linear(512//2,256//2)
        self.linear2 = nn.Linear(640//2,256//2)
        self.linear3 = nn.Linear(256//2,256//2)
        self.linear4 = nn.Linear(384//2,256//2)
        self.linear5 = nn.Linear(512//2,256//2)
        self.linear6 = nn.Linear(512//2,256//2)
        self.linear7 = nn.Linear(640//2,256//2)
        self.linear8 = nn.Linear(256//2,256//2)
        self.linear9 = nn.Linear(384//2,256//2)


        self.attention = Attention_Layer()


        self.linear_1 = nn.Linear(256//2, 512//2)
        self.linear_2 = nn.Linear(256//2, 640//2)
        self.linear_3 = nn.Linear(256//2, 256//2)
        self.linear_4 = nn.Linear(256//2, 384//2)
        self.linear_5 = nn.Linear(256//2, 512//2)
        self.linear_6 = nn.Linear(256//2, 512//2)
        self.linear_7 = nn.Linear(256//2, 640//2)
        self.linear_8 = nn.Linear(256//2, 256//2)
        self.linear_9 = nn.Linear(256//2, 384//2)

        # self.layernorm1 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0.5)
        # # self.dropout2 = nn.Dropout(0.5)
        self.batchnorm2 = nn.BatchNorm1d(32)
        # # self.layernorm2 = nn.LayerNorm(128)

        self.gconv = GAT(128//2)
        # self.dropout3 = nn.Dropout(0.5)





        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64, 64))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_leaky_relu1', nn.LeakyReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(64, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_leaky_relu2', nn.LeakyReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, 2))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(64, 64))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('d_leaky_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(64, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


    def forward(self,data,alpha):#
        x = self.attention1(data)
        x = self.LSTM(x)
        x = self.dropout1(x)


        region1,region2,region3,region4,region5,region6,region7,region8,region9 = Region_apart(x)
        # 将九个区域的维度转换成一致
        out1 = F.leaky_relu(self.linear1(region1))
        out2 = F.leaky_relu(self.linear2(region2))
        out3 = F.leaky_relu(self.linear3(region3))
        out4 = F.leaky_relu(self.linear4(region4))
        out5 = F.leaky_relu(self.linear5(region5))
        out6 = F.leaky_relu(self.linear6(region6))
        out7 = F.leaky_relu(self.linear7(region7))
        out8 = F.leaky_relu(self.linear8(region8))
        out9 = F.leaky_relu(self.linear9(region9))
        out = torch.cat((out1,out2,out3,out4,out5,out6,out7,out8,out9),dim=0)
        out = rearrange(out, '(b i) sl -> b i sl', i=9)


        #out = self.layernorm1(out)
        # 进行注意力计算
        out = self.attention(out)


        # # 经过注意力计算后转成原来的维度
        output1 = F.leaky_relu(self.linear_1(out[:, 0 ,:]))
        output2 = F.leaky_relu(self.linear_2(out[:, 1, :]))
        output3 = F.leaky_relu(self.linear_3(out[:, 2, :]))
        output4 = F.leaky_relu(self.linear_4(out[:, 3, :]))
        output5 = F.leaky_relu(self.linear_5(out[:, 4, :]))
        output6 = F.leaky_relu(self.linear_6(out[:, 5, :]))
        output7 = F.leaky_relu(self.linear_7(out[:, 6, :]))
        output8 = F.leaky_relu(self.linear_8(out[:, 7, :]))
        output9 = F.leaky_relu(self.linear_9(out[:, 8, :]))
        output = torch.cat((output1,output2,output3,output4,output5,output6,output7,output8,output9),dim=1)
        # 将九个脑区合并,32x128

        x = rearrange(output,'b (i sl) -> b i sl',i=32)

        x = self.dropout2(x)
        x = self.batchnorm2(x)

        x = rearrange(x, 'b i sl -> (b i) sl')
        #x = self.layernorm2(x)
        data.x = x
        feature = self.gconv(data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        logits1 = self.class_classifier(feature)

        logits2 = self.domain_classifier(reverse_feature)
        class_output = F.softmax(logits1,dim=1)
        domain_output = F.softmax(logits2,dim=1)

        # return logits,probas

        # class_output = self.class_classifier(feature)
        # domain_output = self.domain_classifier(reverse_feature)

        return logits1,class_output,logits2,domain_output#

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='leaky_relu')
                m.bias.data.zero_()
            elif isinstance(m, GAT):
                torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='leaky_relu')
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, val=0.0)

