import pickle
import torch
import matplotlib.pyplot as plt
import numpy
import os
from utils import load_data, accuracy
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Another way for visualising attention (purely through checkpoint)

def load_object(file_path):
    # Open the file in binary read mode
    with open(file_path, 'rb') as f:
        # Load the object from file
        obj = torch.load(f)
    return obj

def att_score_GAT(h,W,a):
    out_features = a.size(dim=0)/2
    out_features = int(out_features)
    Wh = torch.mm(h, W)


    Wh1 = torch.matmul(Wh, a[:out_features, :])
    Wh2 = torch.matmul(Wh, a[out_features:, :])
    # broadcast add
    e = Wh1 + Wh2.T
    leakyrelu = nn.LeakyReLU(0.2)
    e = leakyrelu(e)
    Wh_att = F.softmax(e, dim=1)
    return Wh_att

def att_score_GATv2(h,W,a):
    out_features = a.size(dim=0)
    in_features = W.size(dim=0)/2
    in_features = int(in_features)
    #print(in_features)
    Wh1 = torch.matmul(h, W[:in_features, :])  # [N, F] @ [F , F'] --> [N, F']
    Wh2 = torch.matmul(h, W[in_features:, :])
    Wh1t = Wh1.unsqueeze(1)  # 2708*1*8
    # Wh2t = Wh2.T             #8*2708 or 7*2708
    Wh2t = Wh2.unsqueeze(0)  # 1*2708*8

    e = Wh1t + Wh2t  # 2708*2708*8 or 2708*2708*7
    # print(e.size())
    leakyrelu = nn.LeakyReLU(0.2)
    e = leakyrelu(e)  # 2708*2708*8 or 2708*2708*7
    # e = torch.einsum('ijk,kb->ij', e, self.a)

    e = torch.matmul(e, a)  # 2708*2708*1
    e = torch.squeeze(e)
    # print(e.size())

    Wh_att = F.softmax(e, dim=1)

    return Wh_att


model1 = load_object('./471_cora_GAT.pkl') #Change this to change model used


#print(model1)
#print(model1['attention_layer_1_head_8.W'])
W1 = model1['attention_layer_1_head_1.W']
a1 = model1['attention_layer_1_head_1.a']
W2 = model1['attention_layer_1_head_2.W']
a2 = model1['attention_layer_1_head_2.a']
W3 = model1['attention_layer_1_head_3.W']
a3 = model1['attention_layer_1_head_3.a']
W4 = model1['attention_layer_1_head_4.W']
a4 = model1['attention_layer_1_head_4.a']
W5 = model1['attention_layer_1_head_5.W']
a5 = model1['attention_layer_1_head_5.a']
W6 = model1['attention_layer_1_head_6.W']
a6 = model1['attention_layer_1_head_6.a']
W7 = model1['attention_layer_1_head_7.W']
a7 = model1['attention_layer_1_head_7.a']
W8 = model1['attention_layer_1_head_8.W']
a8 = model1['attention_layer_1_head_8.a']
W9 = model1['attention_layer_2_head_1.W']
a9 = model1['attention_layer_2_head_1.a']


model2 = load_object('./403_cora_GATv2.pkl')
W12 = model2['attention_layer_1_head_1.W']
a12 = model2['attention_layer_1_head_1.a']
W22 = model2['attention_layer_1_head_2.W']
a22 = model2['attention_layer_1_head_2.a']
W32 = model2['attention_layer_1_head_3.W']
a32 = model2['attention_layer_1_head_3.a']
W42 = model2['attention_layer_1_head_4.W']
a42 = model2['attention_layer_1_head_4.a']
W52 = model2['attention_layer_1_head_5.W']
a52 = model2['attention_layer_1_head_5.a']
W62 = model2['attention_layer_1_head_6.W']
a62 = model2['attention_layer_1_head_6.a']
W72 = model2['attention_layer_1_head_7.W']
a72 = model2['attention_layer_1_head_7.a']
W82 = model2['attention_layer_1_head_8.W']
a82 = model2['attention_layer_1_head_8.a']
W92 = model2['attention_layer_2_head_1.W']
a92 = model2['attention_layer_2_head_1.a']

adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset='cora')

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

device = torch.device("cuda")

features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)

h = features


# att = att_score_GATv2(h,W8,a8)
# print(torch.argmax(att, dim=1))
#
# att = att_score_GATv2(h,W1,a1)
# print(torch.argmax(att, dim=1))

att1 = att_score_GAT(h,W3,a3)
print(torch.argmax(att1, dim=1))
att1 = att1.to('cpu')
att2 = att1.detach().numpy()

# Change this to change layer taken
att2 = att_score_GATv2(h,W32,a32)
print(torch.argmax(att2, dim=1))
att2 = att2.to('cpu')
att2 = att2.detach().numpy()

# Code for plots

n1 = range(3327) # range variable for plots

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Plots of weights for GAT and GATv2')

for i in range(2708):
    l = att1[i, :]

    ax1.plot(n1[0:30], l[0:30])
    l1 = att2[i, :]
    ax2.plot(n1[0:30], l1[0:30])
plt.savefig('att_plot.png')

# Plot part of the attention (heatmap)
att1 = att1[0:30, 0:30]
fig1, (ax3,ax4) = plt.subplots(2)
ax3.imshow(att1, cmap='Reds', aspect='0.2')
# for (j,i),label in numpy.ndenumerate(att1):
#     ax3.text(i,j,numpy.round(label,5),ha='center',va='center')
ax3.set_title('Heatmap of Attention Weights for GAT')


att2 = att2[0:30, 0:30]
ax4.imshow(att2, cmap='Reds', aspect='0.2')
# for (j,i),label in numpy.ndenumerate(att2):
#     ax4.text(i,j,numpy.round(label,5),ha='center',va='center')
ax4.set_title('Heatmap of Attention Weights for GATv2')
plt.savefig('heatmap.png')



