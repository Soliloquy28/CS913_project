# 要在PyTorch中实现Focal Loss，你需要自定义损失函数，以根据原始论文的定义来调整交叉熵损失，
# 使之更加关注困难分类的样本。以下是一个如何在PyTorch中编写Focal Loss的示例：


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss, as described in the paper "Focal Loss for Dense Object Detection".
        
        Args:
        - alpha (float): The weighting factor alpha.
        - gamma (float): The focusing parameter gamma.
        - reduction (str): The method for reducing the loss to a scalar ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss between `inputs` and the ground truth `targets`.
        
        Args:
        - inputs (torch.Tensor): Logits for each class.
        - targets (torch.Tensor): Ground truth labels, where each label is 0 <= targets[i] <= C-1.
        
        Returns:
        - loss (torch.Tensor): The computed focal loss.
        """
        # Compute the softmax over the inputs dimension
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probabilities of the targets class
        pt = torch.exp(-BCE_loss)
        
        # Compute focal loss
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Example usage
if __name__ == '__main__':
    # Assuming `outputs` is the logits from your model's forward pass and `labels` is your ground-truth
    outputs = torch.randn(10, 5, requires_grad=True)  # 10 samples, 5 classes
    labels = torch.randint(0, 5, (10,))
    
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(outputs, labels)
    print("Focal Loss:", loss.item())

    # Backpropagate loss
    loss.backward()


# ### 代码解释：
# - **Focal Loss**：Focal Loss 的目的是减小那些已被分类正确（置信度高）的样本的损失贡献，同时增加难以分类（置信度低）样本的损失贡献。
# - **alpha** 参数用来平衡正负样本的重要性。
# - **gamma** 参数是聚焦参数，它调节难易样本的权重，使模型更加关注于难分类的样本。
# - 通过调用`F.cross_entropy`来计算交叉熵损失，然后根据`gamma`和`alpha`调整每个样本的损失。

# 这个自定义的Focal Loss可以用于训练模型时降低类不平衡问题的影响，特别适用于类别不平衡严重的数据集。