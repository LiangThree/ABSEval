import torch

class PearsonMetric:
    def __init__(self, name='pearson'):
        self.name = name

    def __call__(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)
        #  中心化
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # 计算余弦相似度
        return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))