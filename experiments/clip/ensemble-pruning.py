# Pruning Imports

import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from copy import deepcopy

# from pruning_models import *
from pruning_utils.utils import progress_bar, train, test
from pruning_utils.pruner import pruning_model_random, check_sparsity, pruning_model, prune_model_custom, extract_mask, remove_prune

# ILM VP Imports

from functools import partial
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import io
from PIL import Image

import sys
sys.path.append(".")
from data import prepare_expansive_data, IMAGENETCLASSES, IMAGENETNORMALIZE
from algorithms import generate_label_mapping_by_frequency, get_dist_matrix, label_mapping_base
from tools.misc import *
from tools.mapping_visualization import plot_mapping
from models import ExpansiveVisualPrompt
from cfg import *

# Prepare Data Imports

import os
import json
from collections import OrderedDict
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from data.dataset_lmdb import COOPLMDBDataset

# Misc Imports
import tensorflow as tf
import clip
from tools.clip import get_saparate_text_embedding, DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES
from tools.mapping_visualization import plot_mapping
from models import AdditiveVisualPrompt
from data import prepare_additive_data


p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='flowers102')
args = p.parse_args()

dataset = args.dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loaders, configs = prepare_expansive_data(dataset, data_path=data_path)
# normalize = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])

model, preprocess = clip.load("ViT-B/32")
loaders, class_names = prepare_additive_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess)
templates = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES
txt_emb = torch.cat(get_saparate_text_embedding(class_names, templates, model))
emb_names = np.array([f"T{i//len(class_names)} {class_names[i%len(class_names)]}" for i in range(txt_emb.size(0))])

def network(x):
    x_emb = model.encode_image(x)

    # !!! change
    # x_emb /= x_emb.norm(dim=-1, keepdim=True)
    x_emb_tmp1 = x_emb.clone()
    x_emb_tmp2 = x_emb.clone()
    x_emb = x_emb_tmp1 / x_emb_tmp2.norm(dim=-1, keepdim=True)
    x_emb = x_emb.to(torch.float16)

    logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
    return logits
mapping_network = network

outputs = torch.tensor([], dtype=torch.int16).to(device)
ys = torch.tensor([]).to(device)
p_start = 3
p_end = 4
for p in range(p_start, p_end+1):
    
    # Load LM and VP
    # print(results_path + '/clip_ilm_tp_vp_pruning/' + args.dataset + '/' + str(p) + '0/' + 'seed-7~dataset-' + dataset + '~mapping_interval-1~epoch-200~lr-40~pratio-0.' + str(p) + '000~path-' + str(p) +'0/best.pth')
    best = torch.load(results_path + '/clip_ilm_tp_vp_pruning/' + args.dataset + '/' + str(p) + '0/' + 'seed-7~dataset-' + dataset + '~mapping_interval-1~epoch-200~lr-40~pratio-0.' + str(p) + '000~path-' + str(p) +'0/best.pth')
    mapping_sequence = best['mapping_sequence']
    label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    visual_prompt = AdditiveVisualPrompt(224, 30).to(device)
    visual_prompt.load_state_dict(best['visual_prompt_dict'])
    visual_prompt.eval()

    # Regenerate pruning mask
    model, preprocess = clip.load("ViT-B/32")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    pruning_ratio = p/10
    v_net = model.visual
    if (pruning_ratio != 0):
        pruning_model(v_net, pruning_ratio)
        # current_mask = extract_mask(v_net.mask)
        remove_prune(v_net)

    # Test
    output = torch.tensor([]).to(device)
    total_num = 0
    true_num = 0
    pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing", ncols=100)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        if (p == p_start):
            ys = torch.cat((ys, y), 0)
        with torch.no_grad():
            fx0 = network(visual_prompt(x))
            fx = label_mapping(fx0)
        out = torch.argmax(fx, 1)
        output = torch.cat((output, out), 0)
    print("output: " + str(output))
    total_num = ys.size(0)
    true_num = output.eq(ys).float().sum().item()
    print("output acc: " + str(true_num/total_num))
    if (p == p_start):
        outputs = torch.cat((outputs, output), 0)
        outputs = torch.unsqueeze(outputs, 0)
    else:
        outputs = torch.cat((outputs, torch.unsqueeze(output, 0)), 0)

total_num = ys.size(dim=0)

outputs = torch.transpose(outputs, 0, 1)
outputs = outputs.cpu()
outputs_freq = tf.math.bincount(outputs, axis=-1)
ensemble_outputs = torch.argmax(torch.from_numpy(outputs_freq.numpy()), 1)
# print(ensemble_outputs.size())
ensemble_outputs = ensemble_outputs.to(device)
true_num = ensemble_outputs.eq(ys).float().sum().item()

acc = true_num/total_num

print(acc)
# print("ys: " + str(ys))
# print("outputs: " + str(outputs))
# print("ensemble: " + str(ensemble_outputs))

# Save prediction results
print('Save prediction results')
state_dict = {
    "no_ensemble_members": p_end - p_start + 1,
    "prediction": outputs,
    "ys": ys,
}
fn = dataset + '_ensemble_pruning.pt'
torch.save(state_dict, fn)


