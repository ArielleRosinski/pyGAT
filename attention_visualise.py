import pickle
import torch
import matplotlib.pyplot as plt
import numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_object(file_path):
    # Open the file in binary read mode
    with open(file_path, 'rb') as f:
        # Load the object from file
        obj = torch.load(f)
    return obj

# Loading attention weights for GAT
att1 = load_object('./attention/attention.pt')
print(torch.argmax(att1, dim=1))
att1 = att1.to('cpu')
att1 = att1.detach().numpy()

print(numpy.shape(att1))
print(att1)

n1 = range(2708) # range variable for plots

# Loading attention weights for GATv2
att2 = load_object('./attention/attentionv2.pt')
print(torch.argmax(att2, dim=1))
att2 = att2.to('cpu')
att2 = att2.detach().numpy()
print(att2)
print(numpy.shape(att2))

# Plot part of the attention
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

for i in range(100):
    l = att1[i, :]

    ax1.plot(n1[10:30], l[10:30])
    l1 = att2[i, :]
    ax2.plot(n1[0:30], l1[0:30])
plt.savefig('my_plot.png')

# Plot part of the attention (heatmap)
att1 = att1[2072:2079, 2072:2079]
fig1, (ax3,ax4) = plt.subplots(2)
ax3.imshow(att1, cmap='Reds', aspect='0.2')
for (j,i),label in numpy.ndenumerate(att1):
    ax3.text(i,j,numpy.round(label,5),ha='center',va='center')
ax3.set_title('Heatmap of Attention Weights for GAT')


att2 = att2[0:7, 0:7]
ax4.imshow(att2, cmap='Reds', aspect='0.2')
for (j,i),label in numpy.ndenumerate(att2):
    ax4.text(i,j,numpy.round(label,5),ha='center',va='center')
ax4.set_title('Heatmap of Attention Weights for GATv2')
plt.savefig('heatmap.png')


