import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


def categorical_mapping(categories_dir: str = "categories", col_name: str = "item_id"):
    """
    Loads the categories file and gets the mapping from the original Item ID the index value.

    By convention, the "merlin IDs" are the row number for each item_id in this DataFrame.
    """
    filename = os.path.join(categories_dir, f"unique.{col_name}.parquet")
    df = pd.read_parquet(filename)
    return dict(zip(df[col_name], df.index))


def prepare_embeddings_as_npy_large(input_file: str = "data/interactions_merged_df_timestamp_feature_new_data_with_side_info_10.parquet",
                                    output_file: str = "test/pre-trained-item-id.npy"):
    df = pd.read_parquet(input_file, columns=['item_id', 'description'])
    # Load the vocabulary mapping from 'item_id' strings to integer codes
    with open('data/1226voc_10workflows.json', 'r') as f:
        item_id_mapping = json.load(f)
    
    # Ensure the keys are strings and values are integers
    item_id_mapping = {str(k): int(v) for k, v in item_id_mapping.items()}
    
    df = df.drop_duplicates(subset='item_id', keep='first')

    # Map 'item_id' strings to integer codes using the vocabulary mapping
    df['item_id'] = df['item_id'].map(item_id_mapping)

    client = OpenAI(api_key=API_KEY)
    def get_embedding(text, model="text-embedding-3-large"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    
    tqdm.pandas()
    df['embeddings'] = df.description.progress_apply(lambda x: get_embedding(x, model='text-embedding-3-large')) # 1536 for samll, 3072 for large, but can be reduced to 256

    # keep only the item_id and embeddings columns
    df = df[['item_id', 'embeddings']].reset_index(drop=True)

    df.to_parquet(os.path.join("data", "ids_and_embeddings_new_data_0122.parquet"))
    print(df.head(20))
    
    # reorganize pre-trained embeddings based on the encoded ID.
    mapping = categorical_mapping()
    # map item_id to integers
    df['item_id'] = df['item_id'].map(mapping)

    # sort the DataFrame based on the mapped item_id
    df = df.sort_values(by='item_id')

    # add three additional rows with item_id from 0 to 2 and embeddings as zeros
    additional_rows = pd.DataFrame({
        "item_id": [0, 1, 2],
        "embeddings": [[0] * len(df.iloc[0]['embeddings'])] * 3
    })
    df = pd.concat([additional_rows, df], ignore_index=True)

    # Convert to numpy array
    numpy_array = df['embeddings'].to_numpy()

    # reshape the numpy array to the desired shape
    item_cardinality = len(df)
    print(f"item_cardinality: {item_cardinality}")
    embedding_size = len(df.iloc[0]['embeddings'])
    numpy_array = np.array(list(numpy_array)).reshape((item_cardinality, embedding_size))
    # save the numpy array
    np.save(output_file, numpy_array)


def reduce_openai_embedding_size(path="data/ids_and_embeddings_new_data.parquet", output_file="test/pre-trained-item-id-new-data_1024.npy"):
    df = pd.read_parquet(path)
    # reorganize pre-trained embeddings based on the encoded ID.
    mapping = categorical_mapping()

    # map item_id to integers
    df['item_id'] = df['item_id'].map(mapping)

    # sort the DataFrame based on the mapped item_id
    df = df.sort_values(by='item_id')


    # reduce the size of the embeddings to 1024
    df['embeddings'] = df['embeddings'].apply(lambda x: x[:1024])

    def normalize_l2(x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)
    
    df['embeddings'] = df['embeddings'].apply(normalize_l2)

    # add three additional rows with item_id from 0 to 2 and embeddings as zeros
    additional_rows = pd.DataFrame({
        "item_id": [0, 1, 2],
        "embeddings": [[0] * len(df.iloc[0]['embeddings'])] * 3
    })
    df = pd.concat([additional_rows, df], ignore_index=True)

    # Convert to numpy array
    numpy_array = df['embeddings'].to_numpy()

    # reshape the numpy array to the desired shape
    item_cardinality = len(df)
    embedding_size = len(df.iloc[0]['embeddings'])
    numpy_array = np.array(list(numpy_array)).reshape((item_cardinality, embedding_size))
    # save the numpy array
    np.save(output_file, numpy_array)



if __name__ == "__main__":

    prepare_embeddings_as_npy_large(input_file="data/interactions_merged_df_timestamp_feature_new_data_with_side_info_1226_latest.parquet",
                                    output_file="test/pre-trained-item-id-new-data_0122.npy")
    reduce_openai_embedding_size(path="data/ids_and_embeddings_new_data_0122.parquet", output_file="test/pre-trained-item-id-new-data_0122_1024.npy")
