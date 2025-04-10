import pandas as pd
import torch
import plotly.express as px
from sklearn.manifold import TSNE 
from umap import UMAP
import os

class Visualization:

    def __init__(self, 
                 checkpoint_path="tmp_mlm_bert_base_features_no_LLMembed/checkpoint-440000/pytorch_model.bin", 
                 embedding_key="heads.0.body.0.to_merge.categorical_module.embedding_tables.item_id-list.weight", 
                 label_path="unique.item_id_name_0122.csv"):
        
        self.checkpoint_path = checkpoint_path
        self.embedding_key = embedding_key
        self.label_path = label_path
    
    def extract_embeddings(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f" {self.checkpoint_path} not found")

        state_dict = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))

        if self.embedding_key not in state_dict:
            print("can't find the embedding layers in：")
            print("\n".join(state_dict.keys()))
            raise KeyError(f"please check the embedding layer name, suggest to choose the correct key from the above list")

        embeddings = state_dict[self.embedding_key].numpy()
        print(f"size of the embeddings: {embeddings.shape}")
        return embeddings
    
    def tsne_vis(self):
        embeddings = self.extract_embeddings()

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            metric='cosine',
            random_state=42
        )

        print("start t-SNE downsampling...")
        projected_embeddings = tsne.fit_transform(embeddings)
        print("finished with shape:", projected_embeddings.shape)


        if os.path.exists(self.label_path):

            df = pd.read_csv(self.label_path)

            assert len(df) == embeddings.shape[0], "rows do not match with the number of embeddings"
            hover_text = df['item_id_name'].tolist()
            group_labels = df['classification'].tolist()

        else:
            hover_text = [f"Command {i}" for i in range(embeddings.shape[0])]
            group_labels = ["Unknown"] * embeddings.shape[0]


        fig = px.scatter(
            x=projected_embeddings[:, 0],
            y=projected_embeddings[:, 1],
            labels={'x': 't-SNE-1', 'y': 't-SNE-2'},
            title="Command Embeddings Visualization",
            color=group_labels,
            hover_name=hover_text,
            width=1200,
            height=800
        )

        fig.update_traces(
            marker=dict(
                size=5,
                opacity=0.7,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            selector=dict(mode='markers')
        )

        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            )
        )

        fig.write_html("embedding_visualization_tsne.html")

    def umap_vis(self):

        embeddings = self.extract_embeddings()

        reducer = UMAP(
            n_components=2,
            n_neighbors=200,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )

        print("start UMAP downsampling...")
        projected_embeddings = reducer.fit_transform(embeddings)
        print("finished with shape:", projected_embeddings.shape)

        if os.path.exists(self.label_path):

            df = pd.read_csv(self.label_path)

            assert len(df) == embeddings.shape[0], "rows do not match with the number of embeddings"
            
            hover_text = df['item_id_name'].tolist()
            group_labels = df['classification'].tolist()

        else:
            hover_text = [f"Command {i}" for i in range(embeddings.shape[0])]
            group_labels = ["Unknown"] * embeddings.shape[0]


        fig = px.scatter(
            x=projected_embeddings[:, 0],
            y=projected_embeddings[:, 1],
            labels={'x': 'UMAP-1', 'y': 'UMAP-2'},
            title="Command Embeddings Visualization",
            color=group_labels,
            hover_name=hover_text,
            width=1200,
            height=800
        )

        fig.update_traces(
            marker=dict(
                size=5,
                opacity=0.7,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            selector=dict(mode='markers')
        )

        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            )
        )

        fig.write_html("embedding_visualization_umap.html")


if __name__ == "__main__":
    vis = Visualization()
    vis.tsne_vis()
    vis.umap_vis()
