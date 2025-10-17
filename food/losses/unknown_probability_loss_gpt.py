# import torch
# import torch.distributions as dists
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# import math
# import numpy as np


# class UPLoss(nn.Module):
#     """
#     Unknown Probability Loss for OpenSet目标检测改进版
#     """

#     def __init__(self,
#                  num_classes: int,
#                  sampling_metric: str = "min_score",
#                  sampling_ratio: int = 1,
#                  topk: int = 3,
#                  alpha: float = 1.0,
#                  unk: int = 2,
#                  known_classes_weight: float = 1.0,
#                  unknown_classes_weight: float = 1.0,
#                  dynamic_adjustment_enabled: bool = False):
#         super().__init__()
#         self.num_classes = num_classes
#         assert sampling_metric in ["min_score", "max_entropy", "random", "max_unknown_prob", "max_energy", "max_condition_energy", "VIM", "edl_dirichlet"]
#         self.sampling_metric = sampling_metric
#         self.sampling_ratio = sampling_ratio
#         self.topk = topk
#         self.alpha = alpha
#         self.unk = unk
#         self.known_classes_weight = known_classes_weight
#         self.unknown_classes_weight = unknown_classes_weight
#         self.dynamic_adjustment_enabled = dynamic_adjustment_enabled

#         weight = torch.FloatTensor(1).fill_(0.1)
#         self.weight = nn.Parameter(weight, requires_grad=True)
#         bias = torch.FloatTensor(1).fill_(0)
#         self.bias = nn.Parameter(bias, requires_grad=True)

#         # 用于存储已知类别和未知类别相关的统计信息，以便动态调整
#         self.known_classes_stats = {}
#         self.unknown_classes_stats = {}

#     def _extract_unknown_scores_and_labels(self, scores: Tensor, labels: Tensor):
#         """
#         专门提取未知类别得分和标签的函数
#         """
#         fg_inds = labels!= self.num_classes
#         fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]
#         bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]

#         unknown_fg_scores = fg_scores[:, -1]
#         unknown_bg_scores = bg_scores[:, -1]

#         known_fg_scores = fg_scores[:, :self.num_classes - 1]
#         known_bg_scores = bg_scores[:, :self.num_classes - 1]

#         known_fg_labels = fg_labels
#         known_bg_labels = bg_labels

#         # 生成unknown_labels，这里假设未知类别标签可以通过某种方式确定，比如统一设为一个特定值
#         # 具体如何设置需要根据你的数据和业务逻辑来确定，这里只是示例
#         unknown_labels = torch.full_like(known_fg_labels, self.num_classes)

#         return known_fg_scores, known_bg_scores, known_fg_labels, known_bg_labels, unknown_fg_scores, unknown_bg_scores, unknown_labels
    
#     def _sampling(self, scores: Tensor, labels: Tensor):
#         known_fg_scores, known_bg_scores, known_fg_labels, known_bg_labels, unknown_fg_scores, unknown_bg_scores, _ = self._extract_unknown_scores_and_labels(scores, labels)

#         num_fg = known_fg_scores.size(0)
#         topk = num_fg if (self.topk == -1) or (num_fg <
#                                                self.topk) else self.topk

#         if self.sampling_metric == "max_entropy":
#             pos_metric = dists.Categorical(
#                 known_fg_scores.softmax(dim=1)).entropy()
#             neg_metric = dists.Categorical(
#                 known_bg_scores.softmax(dim=1)).entropy()
#         elif self.sampling_metric == "min_score":
#             pos_metric = -known_fg_scores.max(dim=1)[0]
#             neg_metric = -known_bg_scores.max(dim=1)[0]
#         elif self.sampling_metric == "random":
#             pos_metric = torch.rand(known_fg_scores.size(0),).to(scores.device)
#             neg_metric = torch.rand(known_bg_scores.size(0),).to(scores.device)
#         elif self.sampling_metric == "max_unknown_prob":
#             pos_metric = -unknown_fg_scores
#             neg_metric = -unknown_bg_scores
#         elif self.sampling_metric == "max_energy":
#             pos_metric = -torch.logsumexp(known_fg_scores, dim=1)
#             neg_metric = -torch.logsumexp(known_bg_scores, dim=1)
#         elif self.sampling_metric == "edl_dirichlet":
#             pos_metric = (self.num_classes + 1) / torch.sum(torch.exp(known_fg_scores) + 1, dim=1)
#             neg_metric = (self.num_classes + 1) / torch.sum(torch.exp(known_bg_scores) + 1, dim=1)
#         elif self.sampling_metric == "max_condition_energy":
#             pos_metric = -torch.logsumexp(known_fg_scores, dim=1)
#             neg_metric = -torch.logsumexp(known_bg_scores, dim=1)
#         elif self.sampling_metric == "VIM":
#             known_fg_scores_mean = known_fg_scores - torch.mean(known_fg_scores, dim=0)
#             known_fg_scores_mean_transpose = torch.transpose(known_fg_scores_mean, dim0=1, dim1=0)
#             A = torch.mm(known_fg_scores_mean, known_fg_scores_mean_transpose)
#             A = A / (A.size()[0] - 1)
#             (evals, evecs) = torch.eig(A, eigenvectors=True)  # type: ignore
#             evecs = evecs.detach()
#             pos_metric = - evals[:, 0]
#             _, pos_inds = pos_metric.topk(topk)
#             R = evecs[:, pos_inds]
#             R_transpose = torch.transpose(R, dim0=1, dim1=0)
#             known_fg_scores_transform = torch.mm(R_transpose, known_fg_scores)
#             known_fg_scores = known_fg_scores_transform
#             known_fg_labels = known_fg_labels[pos_inds]
#             neg_metric = -known_bg_scores.max(dim=1)[0]

#         if self.sampling_metric == "VIM":
#             _, neg_inds = neg_metric.topk(topk * self.sampling_ratio)
#             known_bg_scores, known_bg_labels = known_bg_scores[neg_inds], known_bg_labels[neg_inds]
#         else:
#             _, pos_inds = pos_metric.topk(topk)
#             _, neg_inds = neg_metric.topk(topk * self.sampling_ratio)
#             known_fg_scores, known_fg_labels = known_fg_scores[pos_inds], known_fg_labels[pos_inds]
#             known_bg_scores, known_bg_labels = known_bg_scores[neg_inds], known_bg_labels[neg_inds]

#         return known_fg_scores, known_bg_scores, known_fg_labels, known_bg_labels

#     def _soft_cross_entropy(self, A, un_id, mask_scores, targets):
                     
#         targets = targets.sum(dim = 1).unsqueeze(1)
#         logprobs = torch.log(A.unsqueeze(1) + mask_scores)        
#         return -(targets * logprobs).sum() / mask_scores.shape[0]

#     def _calculate_uncertainty(self, scores: Tensor):
#         """
#         计算样本的不确定性，这里简单示例为使用熵来衡量
#         """
#         distribution = dists.Categorical(scores.softmax(dim=1))
#         uncertainty = distribution.entropy()
#         return uncertainty

#     def _hierarchical_loss_calculation(self, known_scores: Tensor, known_labels: Tensor, unknown_scores: Tensor, unknown_labels: Tensor):
#         """
#         层次化损失计算
#         """
#         known_loss = F.cross_entropy(known_scores, known_labels)

#         # 对未知类别计算额外的损失，这里简单示例为与已知类别中心的距离损失
#         known_centroid = torch.mean(known_scores, dim=0)
#         known_centroid = known_centroid.unsqueeze(0).expand((unknown_scores.shape[0], known_centroid.shape[0]))
#         unknown_scores = unknown_scores.unsqueeze(1).expand(-1, known_centroid.shape[1])
#         unknown_distances = torch.norm(unknown_scores - known_centroid, p=2, dim=1)
#         unknown_loss = torch.mean(unknown_distances)

#         return known_loss * self.known_classes_weight + unknown_loss * self.unknown_classes_weight

#     def forward(self, scores: Tensor, labels: Tensor, un_id: Tensor):
#         known_fg_scores, known_bg_scores, known_fg_labels, known_bg_labels = self._sampling(
#             scores, labels)
    
#         # 合并采样后的已知类别分数和标签
#         known_scores = torch.cat([known_fg_scores, known_bg_scores])
#         known_labels = torch.cat([known_fg_labels, known_bg_labels])

#         num_sample, num_classes = known_scores.shape
#         mask = torch.arange(num_classes).repeat(
#             num_sample, 1).to(known_scores.device)
#         known_labels = known_labels.clamp(max = num_classes - 1)
 
#         inds = mask!= known_labels[:, None].repeat(1, num_classes)

#         # 修正索引操作，确保得到的形状正确
#         mask_selected = mask[inds].view(-1)
#         new_num_elements = num_sample * (num_classes - 1)
#         if mask_selected.numel()!= new_num_elements:
#             raise ValueError(f"Expected {new_num_elements} elements after indexing, but got {mask_selected.numel()}")
#         mask = mask_selected.reshape(num_sample, num_classes - 1)

#         gt_scores = torch.gather(
#             F.softmax(known_scores, dim=1), 1, known_labels[:, None]).squeeze(1)
#         mask_scores = torch.gather(known_scores, 1, mask)

#         S = torch.sum(torch.exp(known_scores) + 1, dim=1, keepdim=True)
#         A = self.num_classes / S
#         A = A.squeeze(1)

#         gt_scores[gt_scores < 0] = 0.0
#         targets = torch.zeros_like(mask_scores)
#         num_fg = known_fg_scores.size(0)
#         if self.num_classes > targets.shape[1]:
#             self.num_classes = targets.shape[1]
#         targets[:num_fg, self.num_classes - 2] = gt_scores[:num_fg] * \
#             (1 - gt_scores[:num_fg]).pow(self.alpha)
#         targets[num_fg:, self.num_classes - 1] = gt_scores[num_fg:] * \
#             (1 - gt_scores[:num_fg:]).pow(self.alpha)

#         # 计算不确定性并融入损失计算
#         uncertainty = self._calculate_uncertainty(known_scores)
#         # 调整targets的形状（这里只是示例，可能需要根据实际情况修改）
#         targets = targets.sum(dim = 1).unsqueeze(1)
#         targets = targets * (1 + uncertainty)


#         logprobs = F.log_softmax(mask_scores, dim=1)

        
#         loss = -(targets * logprobs).sum() / mask_scores.shape[0]

        
#         # 正确提取已知类别和未知类别的得分与标签信息
#         known_fg_scores, known_bg_scores, known_fg_labels, known_bg_labels, unknown_fg_scores, unknown_bg_scores, unknown_labels = self._extract_unknown_scores_and_labels(scores, labels) # type: ignore
#         unknown_scores = torch.cat([unknown_fg_scores, unknown_bg_scores])

#         # 层次化损失计算
#         hierarchical_loss = self._hierarchical_loss_calculation(known_scores, known_labels, unknown_scores, unknown_labels)

#         # 根据动态调整机制更新损失
#         if self.dynamic_adjustment_enabled:
#             self._update_dynamic_adjustment(known_scores, known_labels, unknown_scores, unknown_labels)
#             loss = self._adjust_loss_based_on_stats(loss)

#         return loss + hierarchical_loss

#     def _update_dynamic_adjustment(self, known_scores: Tensor, known_labels: Tensor, unknown_scores: Tensor, unknown_labels: Tensor):
#         """
#         根据已知类别和未知类别样本更新相关统计信息，以便动态调整损失
#         """
#         for label in range(self.num_classes - 1):
#             known_inds = known_labels == label
#             if known_inds.sum() > 0:
#                 self.known_classes_stats[label] = {
#                     'mean_score': known_scores[known_inds].mean(),
#                     'entropy': dists.Categorical(known_scores[known_inds].softmax(dim=1)).entropy().mean()
#                 }

#         unknown_inds = unknown_labels == self.num_classes
#         if unknown_inds.sum() > 0:
#             self.unknown_classes_stats[self.num_classes] = {
#                 'mean_score': unknown_scores[unknown_inds].mean(),
#                 'entropy': dists.Categorical(unknown_scores[unknown_inds].softmax(dim=1)).entropy().mean()
#             }

#     def _adjust_loss_based_on_stats(self, loss: Tensor):
#         """
#         根据统计信息动态调整损失
#         """
#         known_weight_adjustment = 1.0
#         unknown_weight_adjustment = 1.0

#         # 这里简单示例根据已知类别平均得分和熵进行调整
#         for label, stats in self.known_classes_stats.items():
#             if stats['mean_score'] < 0.5 and stats['entropy'] > 0.5:
#                 known_weight_adjustment *= 1.5

#         if self.num_classes in self.unknown_classes_stats and self.unknown_classes_stats[self.num_classes]['entropy'] > 0.5:
#             unknown_weight_adjustment *= 1.5

#         return loss * known_weight_adjustment * unknown_weight_adjustment


import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class UPLoss(nn.Module):
    """Unknown Probability Loss for open-set detection.

    Args:
        num_classes (int): Number of classes.
        sampling_metric (str): Metric for sampling. Options include:
            'min_score', 'max_entropy', 'random', 'max_unknown_prob',
            'max_energy', 'max_condition_energy', 'VIM', 'edl_dirichlet'.
        sampling_ratio (int): Ratio for sampling background classes.
        topk (int): Number of top samples to consider.
        alpha (float): Weighting factor for targets.
        unk (int): Index for unknown class.
    """
    def __init__(self,
                 num_classes: int,
                 sampling_metric: str = "min_score",
                 sampling_ratio: int = 1,
                 topk: int = 3,
                 alpha: float = 1.0,
                 unk: int = 2):
        super().__init__()
        self.num_classes = num_classes
        assert sampling_metric in [
            "min_score", "max_entropy", "random", "max_unknown_prob",
            "max_energy", "max_condition_energy", "VIM", "edl_dirichlet"], \
            "Invalid sampling metric specified."
        self.sampling_metric = sampling_metric
        self.sampling_ratio = sampling_ratio
        self.topk = topk if topk != -1 else None  # Handle topk=-1 case
        self.alpha = alpha
        self.unk = unk

        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def _soft_cross_entropy(self, input_gt_scores: Tensor, un_id: Tensor, input: Tensor, target: Tensor) -> Tensor:
        """Calculate soft cross entropy loss."""
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def _sampling(self, scores: Tensor, labels: Tensor) -> tuple:
        """Sample foreground and background scores based on the selected metric."""
        fg_mask = labels != self.num_classes
        fg_scores, fg_labels = scores[fg_mask], labels[fg_mask]
        bg_scores, bg_labels = scores[~fg_mask], labels[~fg_mask]

        # Remove unknown classes
        _fg_scores = torch.cat([fg_scores[:, :self.num_classes - 1], fg_scores[:, -1:]], dim=1)
        _bg_scores = torch.cat([bg_scores[:, :self.num_classes - 1], bg_scores[:, -1:]], dim=1)

        num_fg = fg_scores.size(0)
        topk = min(num_fg, self.topk) if self.topk else num_fg

        # Calculate uncertainties based on the chosen sampling metric
        if self.sampling_metric == "max_entropy":
            pos_metric = dists.Categorical(_fg_scores.softmax(dim=1)).entropy()
            neg_metric = dists.Categorical(_bg_scores.softmax(dim=1)).entropy()
        elif self.sampling_metric == "min_score":
            pos_metric = -_fg_scores.max(dim=1)[0]
            neg_metric = -_bg_scores.max(dim=1)[0]
        elif self.sampling_metric == "random":
            pos_metric = torch.rand(_fg_scores.size(0), device=scores.device)
            neg_metric = torch.rand(_bg_scores.size(0), device=scores.device)
        elif self.sampling_metric == "max_unknown_prob":
            pos_metric = -fg_scores[:, -2]
            neg_metric = -bg_scores[:, -2]
        elif self.sampling_metric in ["max_energy", "max_condition_energy"]:
            pos_metric = -torch.logsumexp(_fg_scores, dim=1)
            neg_metric = -torch.logsumexp(_bg_scores, dim=1)
        elif self.sampling_metric == "edl_dirichlet":
            pos_metric = (self.num_classes + 1) / (torch.sum(torch.exp(fg_scores) + 1, dim=1))
            neg_metric = (self.num_classes + 1) / (torch.sum(torch.exp(bg_scores) + 1, dim=1))
        elif self.sampling_metric == "VIM":
            fg_scores_mean = fg_scores - fg_scores.mean(dim=0)
            A = torch.mm(fg_scores_mean, fg_scores_mean.t()) / (fg_scores.size(0) - 1)
            # evals, evecs = torch.eig(A, eigenvectors=True)
            evals, evecs = torch.linalg.eig(A)
            pos_metric = -evals[:, 0]
            _, pos_inds = pos_metric.topk(topk)
            R = evecs[:, pos_inds]
            fg_scores = torch.mm(R.t(), fg_scores)
            fg_labels = fg_labels[pos_inds]
            neg_metric = -_bg_scores.max(dim=1)[0]

        # Sample topk foreground and background scores
        if self.sampling_metric == "VIM":
            _, neg_inds = neg_metric.topk(topk * self.sampling_ratio)
            bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]
        else:
            _, pos_inds = pos_metric.topk(topk)
            _, neg_inds = neg_metric.topk(topk * self.sampling_ratio)
            fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
            bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]

        return fg_scores, bg_scores, fg_labels, bg_labels

    def forward(self, scores: Tensor, labels: Tensor, un_id: Tensor) -> Tensor:
        """Forward pass for calculating the loss."""
        fg_scores, bg_scores, fg_labels, bg_labels = self._sampling(scores, labels)

        # Combine foreground and background scores
        combined_scores = torch.cat([fg_scores, bg_scores])
        combined_labels = torch.cat([fg_labels, bg_labels])
        num_samples, num_classes = combined_scores.shape

        mask = torch.arange(num_classes, device=scores.device).repeat(num_samples, 1)
        valid_mask = mask != combined_labels[:, None]
        mask = mask[valid_mask].reshape(num_samples, num_classes - 1)

        gt_scores = torch.gather(F.softmax(combined_scores, dim=1), 1, combined_labels[:, None]).squeeze(1)
        mask_scores = torch.gather(combined_scores, 1, mask)

        # Calculate the unknown class scores
        S = torch.sum(torch.exp(combined_scores) + 1, dim=1, keepdim=True)
        A = (self.num_classes / S).squeeze(1)

        gt_scores = F.relu(gt_scores)  # Ensure non-negative scores
        targets = torch.zeros_like(mask_scores)

        num_fg = fg_scores.size(0)
        targets[:num_fg, self.num_classes - 2] = gt_scores[:num_fg] * (1 - gt_scores[:num_fg]).pow(self.alpha)
        targets[num_fg:, self.num_classes - 1] = gt_scores[num_fg:] * (1 - gt_scores[num_fg:]).pow(self.alpha)

        return self._soft_cross_entropy(A, un_id, mask_scores, targets.detach())

        # """
        # 计算unknown类的损失
        # :param scores: 模型预测的得分，形状为 (2048, 22)，其中2048是检测框数量，22是类别数量（包括unknown类）
        # :param gt_classes: 真实类别，形状为 (2048,)，其中2048是检测框数量
        # :param unknown_threshold: 用于判断unknown类的阈值
        # :return: unknown类的损失值
        # """
        # # 获取预测为unknown类的概率
        # unknown_scores = scores[:, -1]  # 假设最后一个类别是unknown类
        # # 创建一个与gt_classes相同形状的掩码，用于标记unknown类
        # is_unknown_mask = (labels == -1).float()  # 假设真实类别中用 -1 表示unknown类

        # # 计算unknown类的损失，这里使用二元交叉熵损失（BCEWithLogitsLoss），将预测得分与unknown类掩码进行比较
        # loss_fn = nn.BCEWithLogitsLoss()
        # loss = loss_fn(unknown_scores, is_unknown_mask)

        # return loss
    

   
    


    