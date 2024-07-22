import torch
import torch.nn as nn
import warnings



warnings.filterwarnings('always')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.cnn_block = nn.Sequential(
#             nn.Conv1d(1024, 512, kernel_size=3, padding=1),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(256, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(128, 64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2)
#             # nn.MaxPool1d(kernel_size=2, stride=2)
#         )
        
#         self.final_conv = nn.Conv1d(64, 4, 1)

#     def forward(self, x):
#         # print("Input shape:", x.shape)
        
#         # Reshape
#         batch_size, channels, length = x.shape
#         x = x.view(batch_size, 1024, 1200)
#         # print("After reshaping:", x.shape)
        
#         # CNN block
#         x = self.cnn_block(x)
#         # print("After CNN:", x.shape)
        
#         # Final convolution
#         x = self.final_conv(x)
#         # print("After final conv:", x.shape)
        
#         # Softmax
#         x = F.softmax(x, dim=1)
        
#         return x



class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, sequence_length=1228800):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            # 初始降采样层
            nn.Conv1d(input_channels, 32, kernel_size=63, stride=128, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # 计算特征提取后的序列长度
        self.feature_length = sequence_length // (128 * 2 * 2 * 2)
        
        self.classifier = nn.Sequential(
            nn.Conv1d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # 输出形状: [batch_size, num_classes, sequence_length]


