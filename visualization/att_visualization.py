import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


command_labels = [
  "Wall",
  "Shape Pane Edit",
  "Modify Layers",
  "Set Active Layer",
  "Wall",
  "Move by Points",
#   "Modify Classes",
  "Send to Front",
  "Modify Classes",
  "Change Class Options",
  "Set Active Class",
  "Modify Classes",
  "Move by Points",
  "Resize",
  "Set Active Class",
  "Rectangle",
  "Resize",
  "Rectangle",
  "Move by Points",
  "Rectangle",
  "Resize",
  "Rectangle",
  "Move by Points",
  "Rectangle",
  "Resize",
  "Rectangle",
  "Resize",
  "Rectangle; Add Surface - ",
  "Set Active Layer",
  "Rectangle",
  "Resize",
  "Change Wall Preferences",
  "Activate Class - ",
  "Shape Pane Edit",
  "Move by Points",
  "Change Class Options",
  "Shape Pane Edit",
  "Move by Points",
  "Modify Classes",
  "Activate Class - ",
  "Shape Pane Edit",
  "Activate Class - ",
  "Line",
  "Delete",
  "Shape Pane Edit",
  "Modify Classes",
  "Move by Points",
  "Shape Pane Edit",
  "Set Active Class",
  "Modify Classes",
  "Set Active Class",
  "Shape Pane Edit",
  "Modify Classes",
  "Move by Points",
  "Resize",
  "Line",
  "Delete",
  "Resize",
  "Delete",
  "Resize",
  "Delete",
  "Resize",
  "Clipping",
  "Resize",
  "Organization - ",
  "Modify Classes",
  "Delete",
  "Shape Pane Edit",
  "Modify Classes",
  "Move by Points",
  "Resize",
  "Delete",
  "Resize",
  "Delete",
  "Move by Points",
  "Resize",
  "Delete",
  "Move by Points",
  "Shape Pane Edit",
  "Modify Classes",
  "Add Surface - ",
  "Activate Class - ",
  "Line",
  "Copy - ",
  "Set Active Layer",
  "Paste In Place - ",
  "Modify Layers",
  "Set Active Layer",
  "Reshape",
  "Organization - "
]

def vis_feature_fusion_map_multi(input="llama_att_weights.pt", output="llama_attn_weights_multi.png", num_samples=9):
    attn_weights = torch.load(input)
    
    labels = [
        "ID",
        "Type",
        "Target",
        "Continuous",
        "Description"
    ]
    
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4))
    

    for i, name in zip(range(num_samples),[0,1,2,3,5,6,7,8,9]):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        weights = attn_weights[i].detach().cpu().numpy()
        
        sns.heatmap(weights, cmap='Greys', annot=True, fmt=".2f", cbar=False, ax=ax)
        ax.set_title(f"{command_labels[name]}")
        ax.set_xlabel("Key Features")
        ax.set_ylabel("Query Features")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels, rotation=0)
    
    if num_samples < n_rows * n_cols:
        for j in range(num_samples, n_rows * n_cols):
            fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()  
    plt.savefig(output)


def vis_feature_att_layers_multi(input_file="llama_layers_att.pt", output_file="llama_layers_att_weights.png"):

    attn_weights = torch.load(input_file)
    layer_1 = attn_weights[0]
    layer_2 = attn_weights[1]

    weights1 = layer_1[0].detach().cpu().numpy()
    weights2 = layer_2[0].detach().cpu().numpy()

    seq_range = (0, 11)
    weights1 = weights1[:, seq_range[0]:seq_range[1], seq_range[0]:seq_range[1]]
    weights2 = weights2[:, seq_range[0]:seq_range[1], seq_range[0]:seq_range[1]]

    weights1 = weights1.mean(axis=0)
    weights2 = weights2.mean(axis=0)

    modified_matrix1 = np.delete(np.delete(weights1, 6, axis=0), 6, axis=1)
    modified_matrix2 = np.delete(np.delete(weights2, 6, axis=0), 6, axis=1)
    
    tick_labels = command_labels[seq_range[0]:6] + command_labels[7:seq_range[1]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.5)

    ax1 = axes[0]
    sns.heatmap(modified_matrix1, cmap='Blues', annot=False, fmt=".2f", ax=ax1)
    ax1.set_title("Attention Weights - llama layer 1")
    ax1.set_xlabel("Key Position")
    ax1.set_ylabel("Query Position")
    ax1.set_xticklabels(tick_labels, rotation=90)
    ax1.set_yticklabels(tick_labels, rotation=0)

    ax2 = axes[1]
    sns.heatmap(modified_matrix2, cmap='Blues', annot=False, fmt=".2f", ax=ax2)
    ax2.set_title("Attention Weights - llama layer 2")
    ax2.set_xlabel("Key Position")
    ax2.set_ylabel("Query Position")
    ax2.set_xticklabels(tick_labels, rotation=90)
    ax2.set_yticklabels(tick_labels, rotation=0)

    plt.tight_layout() 
    plt.savefig(output_file)



if __name__ == '__main__':
    vis_feature_fusion_map_multi()
    vis_feature_att_layers_multi()