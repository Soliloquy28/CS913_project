import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Database import training_dataset_dataloader, validation_dataset_dataloader
from collections import Counter
import matplotlib.pyplot as plt
import gc




def training_part(model, learning_rate, num_epochs, model_name):
    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Class 0: 42.12%
    # Class 1: 46.93%
    # Class 2: 6.81%
    # Class 3: 11.88%
    # 39/43/7/11
    class_weights = torch.tensor([0.0860, 0.0772, 0.5319, 0.3049]).to(device)  # 0: 946235, 1: 1042059, 2: 148641, 3: 263894
    train_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="none")
    val_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    torch.set_grad_enabled(True)


    def train_epoch(dataloader, criterion):
        
        model.train()  
        running_loss = 0.0  
        total = 0

        # 批次处理
        for inputs, labels in tqdm(dataloader):  
            inputs, labels = inputs.to(device), labels.to(device)     

            optimizer.zero_grad()   
            outputs = model(inputs)     
            
            outputs = outputs.permute(0, 2, 1)  # 新形状: [batch_size, sequence_length, num_classes]
            # 这里通过 reshape 将输出和标签展平，以符合 criterion 的输入要求。
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1)).mean()
            
            loss.backward()     
            optimizer.step()    

            # 只考虑非 -1 的标签
            mask = labels != -1
            valid_labels = labels[mask]

            total += valid_labels.size(0)
            running_loss += loss.item() * valid_labels.size(0)
        
        gc.collect()
        # 每个epoch结束后清理一次CUDA缓存
        torch.cuda.empty_cache()

        # 在所有批次处理完成后，计算这一轮训练的平均损失，即总损失除以数据集中的样本总数。
        # 确保total_samples不为0,避免除以0的错误
        epoch_loss = running_loss / total if total > 0 else 0
        
        return epoch_loss


    def validation(dataloader, criterion):

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

 
    best_validation_loss = float('inf')    # 初始化了一个变量 best_val_loss 来记录验证集上观测到的最小损失。float('inf') 表示一开始设置为无限大，以便任何实际观测到的损失值都会更小。
    best_model_path = f'{model_name}_best_model_lr{learning_rate}.pth'
    last_model_path = f'{model_name}_last_model_lr{learning_rate}.pth'

    patience = 10
    trigger_times = 0
        
    training_loss_list = []
    validation_loss_list = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} / {num_epochs}')

        train_epoch_loss = train_epoch(training_dataset_dataloader, train_criterion)
        training_loss_list.append(train_epoch_loss)
        print(f'Training Loss: {train_epoch_loss}.')

        validation_epoch_loss, validation_accuracy, _, _ = validation(validation_dataset_dataloader, val_criterion)
        validation_loss_list.append(validation_epoch_loss)
        print(f'Validation Loss: {validation_epoch_loss}.')
        
        # 早停逻辑
        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss
            trigger_times = 0  # 重置早停计数器
            # 保存最佳模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_validation_loss': best_validation_loss,
                'train_loss': train_epoch_loss,
                'val_loss': validation_epoch_loss,
                'val_accuracy': validation_accuracy,
                'training_loss_list': training_loss_list,
                'validation_loss_list': validation_loss_list
            }
            torch.save(checkpoint, best_model_path)
            print('--------------------------------------Saved best model-------------------------------------------')
        # else:
        #     # print('-------------------------------------------------------------------------------------------------')
        #     trigger_times += 1
        #     if trigger_times >= patience:
        #         print(f"Early stopping triggered at epoch {epoch}!")
        #         break

        torch.cuda.empty_cache()

    # 保存最后一个epoch的模型（可选）
    last_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_epoch_loss,
        'val_loss': validation_epoch_loss,
        'val_accuracy': validation_accuracy,
        'training_loss_list': training_loss_list,
        'validation_loss_list': validation_loss_list,
        'learning_rate': learning_rate
    }
    torch.save(last_checkpoint, last_model_path)

    print("Training completed.")


    # 绘制损失趋势图
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, len(training_loss_list) + 1), training_loss_list, label='Training Loss')
    plt.plot(range(1, len(validation_loss_list) + 1), validation_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 保存图片
    plt.savefig(f'{model_name}_loss_trend_lr{learning_rate}.png')
    print(f"Loss trend graph saved as '{model_name}_loss_trend_lr{learning_rate}.png'")




# # Training batch: 1 (1414)/
# # Testing batch: 5 (129)/

# # 3:
# # Counter({1: 484779, 0: 262623})
# # Validation Loss: 1.1750418487868421, Validation Accuracy: 0.567558020984691.
# # Saved best model

# # 4:




