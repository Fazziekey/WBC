import torch
import torch.nn as nn
import torch.nn.functional as F

class NCELoss(nn.Module):
    def __init__(self, batch_size, num_negative_samples):
        super(NCELoss, self).__init__()
        self.batch_size = batch_size
        self.num_negative_samples = num_negative_samples  # 负样本的数量

    def forward(self, features, labels):
        """
        计算 NCE 损失。
        
        参数:
            features (tensor): 模型的输出特征，维度是 (batch_size, feature_dim)。
            labels (tensor): 数据的真实标签，用于辅助区分正负样本。在某些对比学习设置中，这些可以是不同视图的相同样本的索引。

        返回:
            loss (tensor): NCE损失的值。
        """
        # 获取特征维度大小
        feature_dim = features.size(1)

        # 计算正样本的得分。假设正样本的特征位于 features 的第一列。
        positive_samples = features[:, :1]  # 正样本
        positive_scores = torch.sum(positive_samples * features, dim=1, keepdim=True)  # 正样本的得分 (batch_size, 1)

        # 生成负样本，这里为了简化，我们使用 features 作为负样本的源。
        # 在实践中，负样本通常来自于不同于正样本的数据或数据的随机扰动。
        negative_samples = features[:, 1:]  # 剩下的作为负样本
        negative_scores = torch.matmul(features, negative_samples.t())  # 负样本的得分 (batch_size, num_negative_samples)

        # 将正样本得分和负样本得分结合在一起
        all_scores = torch.cat([positive_scores, negative_scores], dim=1)  # (batch_size, 1 + num_negative_samples)

        # 计算 NCE loss
        loss = -torch.mean(
            torch.log(
                F.softmax(all_scores, dim=1)[:, 0]  # 对所有得分应用 softmax，并获取正样本的概率
            )
        )

        return loss


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, representations):
        """
        计算 InfoNCE 损失。
        
        参数:
            representations (tensor): 形状为 (2 * batch_size, feature_dim) 的张量，
                                      其中包含成对的正样本表示。每一对正样本之间的表示应该是相邻的。

        返回:
            loss (tensor): InfoNCE 损失的值。
        """
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(representations, representations.t())

        # 对角线元素（自我相似度）不应包括在内，因此将其设置为一个大的负值
        similarity_matrix.fill_diagonal_(-1e9)

        # 假设成对的正样本是排列在一起的，即 (i, i+batch_size) 是一对，对于所有 i
        batch_size = representations.shape[0] // 2
        labels = torch.arange(batch_size).to(representations.device)

        # 提取用于正样本对的相似度得分
        positive_similarity = similarity_matrix[:batch_size, batch_size:]

        # 计算 InfoNCE loss
        loss = F.cross_entropy(positive_similarity / self.temperature, labels)

        return loss
