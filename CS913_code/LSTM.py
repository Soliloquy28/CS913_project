import torch
import torch.nn as nn
import torch.nn.functional as F


# class LSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pre_len=4, dropout_rate=0.3):
#         super(LSTM, self).__init__()
#         self.pre_len = pre_len
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
        
#         self.model1 = nn.Sequential(
        	
#             nn.Conv1d(input_dim, 256, 1),
#             nn.BatchNorm1d(256),
#             # nn.Conv1d(22, 64, 1),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             # nn.MaxPool1d(2),  
            
#         )

#         self.model2 = nn.LSTM(256, hidden_dim, num_layers=self.num_layers, batch_first=True)
#         self.lstm_dropout = nn.Dropout(dropout_rate)
        
#         self.model3 = nn.Sequential(
#             nn.Linear(in_features=hidden_dim, out_features=64, bias=True),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(in_features=64, out_features=output_dim, bias=True),
#         )


#     def forward(self, x):
#         x = self.model1(x)
#         x = x.transpose(1, 2)  # 调整维度以适应LSTM输入
#         x, _ = self.model2(x)
#         x = self.lstm_dropout(x)
#         x = x[:, -1, :]  # 只取LSTM最后一个时间步的输出
#         x = self.model3(x)
#         return x


# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=4):
#         super(LSTM, self).__init__()
        
#         self.downsample = nn.Sequential(
#             nn.Conv1d(input_size, 32, kernel_size=50, stride=32, padding=25),
#             nn.LeakyReLU(),
#             nn.Conv1d(32, 32, kernel_size=25, stride=32, padding=12),
#             nn.LeakyReLU()
#         )
        
#         self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, 
#                             num_layers=num_layers, batch_first=True)
        
#         self.fc = nn.Linear(hidden_size, output_size)
        
#     def forward(self, x):
#         print("Input shape:", x.shape)
#         # x = x.transpose(1, 2)
#         # print("After first transpose:", x.shape)
#         x = self.downsample(x)
#         print("After downsample:", x.shape)
#         # x = self.adaptive_pool(x)
#         # print("After adaptive pool:", x.shape)
#         x = x.transpose(1, 2)
#         print("Before LSTM:", x.shape)
#         x, _ = self.lstm(x)
#         print("After LSTM:", x.shape)
#         x = self.fc(x)
#         print("Final output:", x.shape)
#         return x


class LSTM(nn.Module):
    def __init__(self, input_features=1024, hidden_dim=128, num_layers=2, num_classes=4):
        super(LSTM, self).__init__()
        # LSTM层，batch_first=True表示输入输出的维度为(batch, seq, feature)
        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # print("Input shape:", x.shape)
        # 调整输入维度：从 [batch_size, 1, 1228800] 到 [batch_size, 1200, 1024]
        # Window reshape into 1200x128
        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels, 1200, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, 1200)
        # print("After reshaping:", x.shape)  

        # 准备 LSTM 输入
        x = x.transpose(1, 2)  # 形状变为 [batch_size, 1200, 32]
        # print("After transpose:", x.shape)

        # 通过LSTM处理
        x, _ = self.lstm(x)  # 输出形状为[batch_size, seq_len, hidden_dim]
        # print("After LSTM:", x.shape)

        # 将LSTM输出通过全连接层转换为类别预测
        x = self.fc(x)  # 输出形状为[batch_size, seq_len, num_classes]

        # 转置以匹配所需的输出形状
        x = x.transpose(1, 2)  # 形状变为 [batch_size, 4, 1200]
        # print("Final output:", x.shape)

        # 应用softmax得到最终的分类概率
        x = F.softmax(x, dim=-1)  # 沿着类别的维度应用Softmax

        return x