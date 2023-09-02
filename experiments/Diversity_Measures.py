# Pruning Imports

import sys
sys.path.append('.')
import torch
import os
import argparse

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='flowers102')
p.add_argument('--ensemble', choices=["pruning", "piece", "seed"], default='pruning')
args = p.parse_args()

dataset = args.dataset
ensemble = args.ensemble

fn = dataset + '_ensemble_' + ensemble + '.pt'
pred_results = torch.load(fn)

no_ensemble_members = pred_results['no_ensemble_members']
prediction = pred_results['prediction']
prediction = prediction.transpose(0,1).to(device)
ys = pred_results['ys']
ys = ys.unsqueeze(0).to(device)
# print('prediction')
# print(prediction)
# print('ys')
# print(ys)
print('no of test sampeles')
print(ys.size(1))
# print('no_ensemble_members')
# print(no_ensemble_members)
# print('ys.size()')
# print(ys.size())
# print('prediction.size()')
# print(prediction.size())

pred_true_num = prediction.eq(ys).float().sum(1)
print('pred_true_num')
print(pred_true_num)

pred_eval = prediction.eq(ys).float().to(device)
ones = torch.ones(pred_eval.size()).to(device)
pred_eval_neg = ones - pred_eval
print('pred_eval')
print(pred_eval)
print('pred_eval_neg')
print(pred_eval_neg)

N11 = (pred_eval * pred_eval[0,:]).sum(1).unsqueeze(0)
N01 = (pred_eval_neg * pred_eval[0,:]).sum(1).unsqueeze(0)
N10 = (pred_eval * pred_eval_neg[0,:]).sum(1).unsqueeze(0)
N00 = (pred_eval_neg * pred_eval_neg[0,:]).sum(1).unsqueeze(0)
for i in range(1, no_ensemble_members):
    N11 = torch.cat((N11, (pred_eval * pred_eval[i,:]).sum(1).unsqueeze(0)), 0)
    N01 = torch.cat((N01, (pred_eval_neg * pred_eval[i,:]).sum(1).unsqueeze(0)), 0)
    N10 = torch.cat((N10, (pred_eval * pred_eval_neg[i,:]).sum(1).unsqueeze(0)), 0)
    N00 = torch.cat((N00, (pred_eval_neg * pred_eval_neg[i,:]).sum(1).unsqueeze(0)), 0)

print('N11')
print(N11)
print('N01')
print(N01)
print('N10')
print(N10)
print('N00')
print(N00)

CK = 2 * ((N11 * N00) - (N01 * N10)) / ((N11 + N10) * (N01 + N00) + (N11 + N01) * (N10 + N00))
print('CK')
print(CK)
Q_CK = (CK.sum()-torch.diagonal(CK,0).sum())/2 * (2/no_ensemble_members/(no_ensemble_members-1))
print('Q_CK')
print(Q_CK)

QS = ((N11 * N00) - (N01 * N10)) / ((N11 * N00) + (N01 * N10))
print('QS')
print(QS)
Q_QS = (QS.sum()-torch.diagonal(QS,0).sum())/2 * (2/no_ensemble_members/(no_ensemble_members-1))
print('Q_QS')
print(Q_QS)

BD = (N01 + N10) / (N11 + N01 + N10 + N00)
print('BD')
print(BD)
Q_BD = (BD.sum()-torch.diagonal(BD,0).sum())/2 * (2/no_ensemble_members/(no_ensemble_members-1))
print('Q_BD')
print(Q_BD)

# print(CK.sum())
# print(torch.diagonal(CK,0).sum())

# state_dict = {
#     "no_ensemble_members": p_end - p_start + 1,
#     "prediction": outputs,
#     "ys": ys,
# }



