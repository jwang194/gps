import torch
from functools import reduce

def prs_cov(prs,device):
    def loss(C,idx):
        n,c = C.shape
        C1 = C - C.mean(0)
        C2 = C1 / C.std(0)
        sub_prs = [p[idx] for p in prs]
        P = lambda x: x - (C2 @ (torch.inverse(C2.T @ C) @ (C2.T @ x)))
        sub_prs = [P(p) for p in sub_prs]
        total_prs = reduce(lambda x,y: x + y,sub_prs,torch.zeros_like(sub_prs[0]).to(device))
        return torch.abs(torch.var(total_prs) - torch.sum(torch.stack([p.var() for p in sub_prs])))/(torch.var(total_prs))
    return loss
