import torch
import torch.nn as nn
from tqdm import tqdm
from Database import testing_dataset_dataloader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from collections import Counter


def testing_part(model, learning_rate, model_name):
    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    torch.set_grad_enabled(True)


    def validation(dataloader):

            model.eval()   
            running_loss = 0.0
            correct = 0
            total = 0
            predicted_labels = []
            true_labels = []
            
            with torch.no_grad():  

                for inputs, labels in tqdm(dataloader):

                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    outputs = outputs.permute(0, 2, 1)  # 新形状: [batch_size, sequence_length, num_classes]
                    loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1)).mean()

                    # 只考虑非 -1 的标签
                    mask = labels != -1
                    valid_outputs = outputs[mask]
                    valid_labels = labels[mask]

                    predicted = valid_outputs.argmax(1)  
                    correct += predicted.eq(valid_labels).sum().item()    
                    total += valid_labels.size(0)     
                    running_loss += loss.item() * valid_labels.size(0)
                
                    predicted_list = predicted[:].tolist()
                    predicted_labels.extend(predicted_list)
                    true_labels.extend(valid_labels[:].tolist())
                
                gc.collect()
                # 每个epoch结束后清理一次CUDA缓存
                torch.cuda.empty_cache()

            count = Counter(predicted_labels)
            print("Total samples:", total)
            print("Correct predictions:", correct)
            # print("Predicted labels:", predicted_labels[:1000])
            print(count)

            epoch_loss = running_loss / total if total > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0

            return epoch_loss, epoch_accuracy, predicted_labels, true_labels


    # 加载最佳模型
    checkpoint = torch.load(f'{model_name}_best_model_lr{learning_rate}.pth')  # or 'last_model_lr{learning_rate}.pth'
    # checkpoint = torch.load(f'best_model_lr{learning_rate}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 进行最终测试
    test_loss, test_accuracy, predicted_labels, true_labels = validation(testing_dataset_dataloader)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)

    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建注释数组
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.2f}%)'
            # annot[i, j] = f'{cm_percent[i, j]:.2f}%'

    # 定义新的标签
    labels = ['Wake', 'Light', 'Deep', 'REM']

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 10))
    # sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', annot_kws={'size': 16}, xticklabels=labels, yticklabels=labels, cbar=False)
    # sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', annot_kws={'size': 16}, xticklabels=labels, yticklabels=labels)
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues', annot_kws={'size': 16}, xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name}: Confusion Matrix (acc={format(test_accuracy, '.4f')})", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.savefig(f'{model_name}_confusion_matrix_lr{learning_rate}.png')
    plt.close()


    # 输出分类报告
    # class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']  # 替换为你的实际类别名称
    report = classification_report(true_labels, predicted_labels)
    print("Classification Report:")
    print(report)

    # 假设 true_labels 和 predictions 是你之前得到的真实标签和预测标签
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    print(f'Cohen\'s Kappa: {kappa}.')

    print(f'Accuracy: {str(test_accuracy)}')

    # 保存分类报告到文件
    with open(f'{model_name}_classification_report_lr{learning_rate}.txt', 'w') as f:
        f.write(report)
        f.write(f'\nCohen\'s Kappa: {str(kappa)}\n')
        f.write(f'Accuracy: {str(test_accuracy)}')

    print('Finished writing results into txt file.')
