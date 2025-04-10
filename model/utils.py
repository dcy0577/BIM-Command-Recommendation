import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)
        return control

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def augment_data_with_llm_side_info(parquet_data, llm_side_info_csv, output_file):
    # this is essentially a join operation
    df_parquet = pd.read_parquet(parquet_data)
    df_llm_csv = pd.read_csv(llm_side_info_csv)

    df_merged = df_parquet.merge(
    df_llm_csv[['message_content', 'summary', 'classification', 'target']],
    left_on='item_id',
    right_on='message_content',
    how='left'
    )
    df_merged.rename(columns={
        'summary': 'description',
    }, inplace=True)

    df_merged.drop(columns='message_content', inplace=True)

    df_merged.to_parquet(output_file, index=False)

    return df_merged



def split_sessions_with_correlated_ids(data, min_items=50, max_items=100, output_file="data/interactions_merged_df_timestamp_feature_augmented_new_data.parquet"):
    np.random.seed(1)
    new_sessions = []
    grouped = data.groupby('session_id')
    for session_id, group in tqdm(grouped, desc="Splitting sessions"):
        split_counter = 1
        while len(group) >= min_items:
            remaining_items = len(group)
            if remaining_items <= max_items:
                chunk_size = remaining_items
            elif remaining_items - min_items < min_items:
                chunk_size = remaining_items
            else:
                chunk_size = np.random.randint(min_items, max_items + 1)
            chunk = group.iloc[:chunk_size]
            group = group.iloc[chunk_size:]
            new_session_id = f"{session_id}-{split_counter}"
            chunk = chunk.copy()
            chunk['session_id'] = new_session_id
            split_counter += 1
            new_sessions.append(chunk)
        # we will not drop any remaining items, but the last session will probably have more items than the max_items
        # theoretically, the lentgh of the last session should be between [min_items, max_items + min_items)
        if 0 < len(group) < min_items:
            last_chunk = new_sessions[-1]
            last_chunk = pd.concat([last_chunk, group], ignore_index=True)
            new_sessions[-1] = last_chunk
    augmented_data = pd.concat(new_sessions, ignore_index=True).reset_index(drop=True)
    augmented_data.to_parquet(output_file)
    return augmented_data


def split_sessions_not_random(data, min_items=5, max_items=200, output_file="data/interactions_merged_df_timestamp_feature_augmented_new_data.parquet"):
    new_sessions = []
    grouped = data.groupby('session_id')
    for session_id, group in tqdm(grouped, desc="Splitting sessions"):
        split_counter = 1
        while len(group) >= min_items:
            remaining_items = len(group)
            if remaining_items <= max_items:
                chunk_size = remaining_items
            else:
                chunk_size = max_items
            chunk = group.iloc[:chunk_size]
            group = group.iloc[chunk_size:]
            new_session_id = f"{session_id}-{split_counter}"
            chunk = chunk.copy()
            chunk['session_id'] = new_session_id
            split_counter += 1
            new_sessions.append(chunk)
        # we will not drop any remaining items, but the last session will probably have more items than the max_items
        # theoretically, the lentgh of the last session should be between [min_items, max_items + min_items)
        if 0 < len(group) < min_items:
            last_chunk = new_sessions[-1]
            last_chunk = pd.concat([last_chunk, group], ignore_index=True)
            new_sessions[-1] = last_chunk
    augmented_data = pd.concat(new_sessions, ignore_index=True).reset_index(drop=True)
    augmented_data.to_parquet(output_file)
    return augmented_data

def save_random_splits_gpu(
    data,
    output_dir: str,
    timestamp_col: str = "ts/first",
    test_size: float = 0.1,
    val_size: float = 0.1,
    overwrite: bool = True,
):
    """Split a dataset into train/val/test splits by randomly shuffling the data.
    Note, this function requires Rapids dependencies to be installed:
    cudf, cupy, and dask_cudf.
    Parameters
    -----
    data: Union[merlin.io.dataset.Dataset, dask_cudf.DataFrame]
        Dataset to split into random splits.
    output_dir: str
        Output path to save the splits.
    timestamp_col: str
        Timestamp column to use to sort the data.
    test_size: float
        Size of the test split, between 0.0 and 1.0.
    val_size: float
        Size of the validation split, between 0.0 and 1.0.
    overwrite: bool
        Whether or not to overwrite the output_dir if it already exists.
    """
    import os
    import shutil
    import cupy
    import dask_cudf
    from merlin.io.dataset import Dataset

    if isinstance(data, dask_cudf.DataFrame):
        data = Dataset(data)
    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Convert to Dask DataFrame and compute
    df = data.to_ddf().compute()
    df = df.sort_values(timestamp_col)

    cupy.random.seed(1)
    random_values = cupy.random.rand(len(df))
    train_size = 1 - val_size - test_size

    if train_size < 0:
        raise ValueError("train_size cannot be negative.")

    # Randomly split the data
    train_set = df[random_values <= train_size]
    valid_set = df[
        (random_values > train_size) & (random_values <= (train_size + val_size))
    ]
    test_set = df[random_values > (1 - test_size)]

    # Save the splits
    os.makedirs(output_dir, exist_ok=True)
    train_set.to_parquet(os.path.join(output_dir, "train.parquet"))
    valid_set.to_parquet(os.path.join(output_dir, "valid.parquet"))
    test_set.to_parquet(os.path.join(output_dir, "test.parquet"))


def save_random_splits_cpu(
    data,
    output_dir: str,
    timestamp_col: str = "ts/first",
    test_size: float = 0.1,
    val_size: float = 0.1,
    overwrite: bool = True,
):
    """Split a dataset into train/val/test splits by randomly shuffling the data.
    Note, this function requires Rapids dependencies to be installed:
    cudf, cupy, and dask_cudf.
    Parameters
    -----
    data: Union[merlin.io.dataset.Dataset, dask_cudf.DataFrame]
        Dataset to split into random splits.
    output_dir: str
        Output path to save the splits.
    timestamp_col: str
        Timestamp column to use to sort the data.
    test_size: float
        Size of the test split, between 0.0 and 1.0.
    val_size: float
        Size of the validation split, between 0.0 and 1.0.
    overwrite: bool
        Whether or not to overwrite the output_dir if it already exists.
    """
    import os
    import shutil

    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # compute dask dataframe
    df = data.compute()
    df = df.sort_values(timestamp_col)

    np.random.seed(1)
    random_values = np.random.rand(len(df))
    train_size = 1 - val_size - test_size

    if train_size < 0:
        raise ValueError("train_size cannot be negative.")

    # Randomly split the data
    train_set = df[random_values <= train_size]
    valid_set = df[
        (random_values > train_size) & (random_values <= (train_size + val_size))
    ]
    test_set = df[random_values > (1 - test_size)]

    # Save the splits
    os.makedirs(output_dir, exist_ok=True)
    train_set.to_parquet(os.path.join(output_dir, "train.parquet"))
    valid_set.to_parquet(os.path.join(output_dir, "valid.parquet"))
    test_set.to_parquet(os.path.join(output_dir, "test.parquet"))

if __name__ == "__main__":

    data = pd.read_parquet("data/interactions_merged_df_timestamp_feature_new_data.parquet")
    # store data in dataframes
    data_df = pd.DataFrame(data)
    # Apply the function to the dataset
    augmented_dataset_with_ids = split_sessions_with_correlated_ids(data_df, output_file="data/interactions_merged_df_timestamp_feature_augmented_new_data.parquet")