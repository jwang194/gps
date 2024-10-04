import os
import sys
import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F

from optparse import OptionParser
from torch import optim
from torch.utils.data import DataLoader

from modules import PopStruct
from loss import prs_cov

parser = OptionParser()
parser.add_option("--pheno",
    dest="pheno",
    type=str,
    default=None,
    help="path to a pair of files, containing the training and test FIDs, IIDs, and phenotypes"
)
parser.add_option("--pgs",
    dest="pgs",
    type=str,
    default=None,
    help="path to a directory containing files with per-chromosome PGS for the phenotype, the sscore outputs of PLINK2 --score"
)
parser.add_option("--pcs",
    dest="pcs",
    type=str,
    default="../gwas/data/covs/all_pcs.qcovar",
    help="path to a file containing FIDs, IIDs, and PCs to use as input for the model"
)
parser.add_option("--lr",
    dest="lr",
    type=float,
    default=2e-3,
    help="learning rate of model parameters",
)
parser.add_option("--batch_size",
    dest="bs",
    type=int,
    default=10**5,
    help="batch size"
)
parser.add_option("--epochs",
    dest="epochs",
    type=int,
    default=10**3,
    help="number of training epochs"
)
parser.add_option("--which_cuda",
    dest="wdev",
    type=int,
    default=0,
    help="which cuda device to use"
)
parser.add_option("--out",
    dest="outdir",
    type=str,
    default=None,
    help="output directory"
)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)

device = torch.device('cuda:%i'%opt.wdev if torch.cuda.is_available() else 'cpu')

if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
    os.makedirs(opt.outdir+'weights')

train_phenotypes,test_phenotypes = [pd.read_csv(opt.pheno+'%s.txt.gz'%s,sep='\t') for s in ['TRAIN','TEST']]
train_phenotypes.index,test_phenotypes.index = train_phenotypes.IID,test_phenotypes.IID
phenotypes = [train_phenotypes,test_phenotypes]
train_n,test_n = [p.shape[0] for p in phenotypes]
N = [train_n,test_n]
files = os.listdir(opt.pgs)
scores = []
for f in files:
    if '.sscore.gz' in f:
         scores.append(pd.read_csv(opt.pgs+f,sep='\t'))

all_scores = scores[0]
all_scores['SCORE1'] = all_scores.ALLELE_CT*all_scores.SCORE1_AVG
for s in range(1,len(scores)):
    all_scores['SCORE%i'%(s+1)] = scores[s].ALLELE_CT*scores[s].SCORE1_AVG

all_scores.index = all_scores.IID
train_scores,test_scores = [p.merge(all_scores,left_index=True,right_index=True) for p in phenotypes]
scores = [train_scores,test_scores]

all_pcs = pd.read_csv(opt.pcs,sep='\t')
all_pcs.index = all_pcs.IID
pc_columns = all_pcs.columns[2:]
train_pcs,test_pcs = [torch.tensor(np.array(s.merge(all_pcs,left_index=True,right_index=True)[pc_columns])) for s in scores]
pcs = [train_pcs,test_pcs]

train_prs_cov,test_prs_cov = [prs_cov([torch.tensor(np.array(s[['SCORE%i'%i]])) for i in range(1,23)],device) for s in scores]
covs = [train_prs_cov,test_prs_cov]

# train_loader = DataLoader([(train_scores[i],train_pcs[i]) for i in range(train_scores.shape[0])],batch_size=opt.bs,shuffle=True)

model = PopStruct(4,train_pcs.shape[1],train_pcs.shape[1]*64,1)

model_optimizer = optim.Adam(model.parameters(),lr=opt.lr)
model.train()

lmd = 1
log = []
log.append('epoch %i: %.4f, %.4f'%tuple([-1]+[c(p,list(range(n))).item() for c,p,n in zip(covs,pcs,N)]))
for epoch in range(opt.epochs):
    model_optimizer.zero_grad()
    train_loss,test_loss = [c(p,list(range(n))) for c,p,n in zip(covs,pcs,N)]
    l = 'epoch %i: %.4f, %.4f'%(epoch,train_loss.item(),test_loss.item())
    train_loss += lmd*torch.abs(list(model.parameters())[0]).sum()
    train_loss.backward()
    model_optimizer.step()
    print(l)
    log.append(l)
    np.savetxt(opt.outdir+'log.txt',log,'%s')
    torch.save(model.state_dict(),opt.outdir+'weights/weights.%s.pt'%str(epoch).zfill(5))
