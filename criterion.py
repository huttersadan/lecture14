import torch
import torch.nn as nn

# TODO 7 可选, 调用库函数或自己额外实现loss和正确率
# 这可能跟你网络输出的维度，以及输出的是logit还是预测概率有关
# (但还是建议网络直接的输出值是logit, 因为通常来说分类loss都需要计算熵, 用log_softmax等方法可以提高数值计算精度，在类别数很多的任务的训练初期可能会产生比较大的影响)
class CrossEntropyWithLogit(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logit:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        B, C = logit.shape
        log_prob = logit.log_softmax(dim=-1)
        entropy_to_add = -log_prob[torch.arange(B), target]
        loss = entropy_to_add.mean()
        return loss

class Accuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logit:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        B, C = logit.shape
        pred = logit.argmax(dim=-1)
        acc = torch.sum(pred == target) / float(target.numel())
        return acc