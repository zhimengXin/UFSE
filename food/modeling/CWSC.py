import torch
import torch.nn as nn

class CWSC(nn.Module):
    def __init__(self, num_classes, weight_dim, sparsity_prob, temperature):
        super(CWSC, self).__init__()
        self.num_classes = num_classes
        self.weight_dim = weight_dim
        self.sparsity_prob = sparsity_prob
        self.temperature = temperature
    #def __init__(self, num_classes, weight_dim, sparsity_prob, temperature):    
        self.weights = nn.Parameter(torch.randn(weight_dim, num_classes)) # 初始化权重        
        self.retention_prob = 1 - sparsity_prob # 计算保留概率

    def forward(self, x):        
        normalized_weights = self.weights / self.weights.norm(dim=0) # 归一化权重
        # 根据保留概率生成随机掩码
        mask = torch.bernoulli(torch.ones_like(normalized_weights) * self.retention_prob)
        # 应用掩码进行权重稀疏化
        masked_weights = mask * normalized_weights
        # 计算类logit输出
        logits = self.temperature * x.matmul(masked_weights)
        return logits
    
    # 假设输入特征维度为1024，类别数量为20（包括已知类和未知类），稀疏概率为0.6，温度因子为20
num_classes = 20
weight_dim = 1024
sparsity_prob = 0.6
temperature = 20

# 创建CWSC实例
cwc = CWSC(num_classes, weight_dim, sparsity_prob, temperature)

# 生成一个随机输入特征（假设批次大小为16）
input_features = torch.randn(16, 1024)

# 通过CWSC计算类logit输出
output_logits = cwc(input_features)
print(output_logits.shape)  