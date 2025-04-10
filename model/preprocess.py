import json
import os
import shutil
import random
import pyarrow as pa
import pandas as pd
import dask.dataframe as dd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags
from model.utils import split_sessions_with_correlated_ids, augment_data_with_llm_side_info


def prepare_df_for_preprocessing_already_merged(path="data/1226_workflow_united_dropped_log_with_id.parquet", 
                                                output_path="data/interactions_merged_df_timestamp_feature_1226_latest.parquet"):
    
    full_data = pd.read_parquet(path, columns=['session_anonymized', 'ts', 'message_content', 'merge_count', 'cat'])
    # print(full_data.head())
    
    # rename the columns
    full_data.rename(columns={
        'session_anonymized': 'session_id',
        'message_content': 'item_id',
        'ts': 'timestamp'
    }, inplace=True)

    # calculate the interval between timestamps within each session, create new column 'timestamp_interval'
    full_data['timestamp_interval'] = full_data.groupby('session_id')['timestamp'].diff().fillna(0)

    full_data.dropna(inplace=True)

    full_data.to_parquet(output_path)

    print("Parquet file saved!")
    pd.set_option('display.max_columns', None)
    print(full_data.tail(20))



def preprocessing(inter_data_path, 
                  data_path="data/processed_nvt_new_data_10",
                  workflow_path="data/workflow_etl_new_data_10"):

    # Encodes categorical features as contiguous integers
    cat_feats = ColumnSelector(['item_id']) >> nvt.ops.Categorify(on_host=True)
    class_feats = ColumnSelector(['classification']) >> nvt.ops.Categorify(on_host=True)
    target_feats = ColumnSelector(['target']) >> nvt.ops.Categorify(on_host=True)
    command_cat_feats = ColumnSelector(['cat']) >> nvt.ops.Categorify(on_host=True)

    # normalize timestamp interval
    timestamp_interval= ColumnSelector(['timestamp_interval'])
    timestamp_interval_norm_global = (
        timestamp_interval >> 
        nvt.ops.Normalize() >> 
        nvt.ops.Rename(name = 'timestamp_interval_norm_global')
    )

    # normalize the item merge count
    merge_count = (
    ColumnSelector(['merge_count']) >>
    nvt.ops.LambdaOp(lambda col: col.astype('float32'))
    )
    merge_count_norm = (
        merge_count >> 
        nvt.ops.Normalize() >> 
        nvt.ops.Rename(name = 'merge_count_norm')
    )

    time_features = (
    timestamp_interval_norm_global +
    merge_count_norm +
    merge_count
    )

    features = ColumnSelector(['session_id', 'timestamp', 'timestamp_interval']) + cat_feats + time_features + command_cat_feats + class_feats + target_feats 

    groupby_features = features >> nvt.ops.Groupby(
        groupby_cols=["session_id"], 
        sort_cols=["timestamp"],
        aggs={
            'item_id': ["list", "count"],
            'timestamp': ["list", "first"],
            'event_time_dt': ["first"],
            'timestamp_interval': ["list"],
            'timestamp_interval_norm_global': ["list"],
            'classification': ["list", "count"],
            'target': ["list", "count"],
            'merge_count': ["list"],
            'merge_count_norm': ["list"],
            'cat': ["list"],
            },
        name_sep="-")


    item_feat = groupby_features['item_id-list'] >> nvt.ops.TagAsItemID()

    timestamp_feat = groupby_features['timestamp_interval-list'] >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
    timestamp_feat += groupby_features['timestamp_interval_norm_global-list'] >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
    category_feat = groupby_features['classification-list'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
    target_feat = groupby_features['target-list'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
    timestamp_feat += groupby_features['merge_count-list'] >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
    timestamp_feat += groupby_features['merge_count_norm-list'] >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
    timestamp_feat += groupby_features['cat-list'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])

    groupby_features_list =  item_feat  + timestamp_feat + category_feat + target_feat

    # TODO: What is this for, is it necessary? Seems like removing it will cause the error
    # now with the new split_sessions_with_correlated_ids the SESSIONS_MAX_LENGTH will theoritcally be 100+5, so 200 is safe
    SESSIONS_MAX_LENGTH = 200
    groupby_features_truncated = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH)


    # tag session_id column for serving with legacy api
    sess_id = groupby_features['session_id'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])

    # Select features for training 
    selected_features = sess_id + groupby_features['item_id-count'] + groupby_features_truncated

    # Filter out sessions with less than 5 interactions 
    MINIMUM_SESSION_LENGTH = 5
    filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] > MINIMUM_SESSION_LENGTH) 

    # Define the NVTabular dataset
    # this will cause cu:400: Total number of concatenated chars exceeds size_type range, this is because the item_id is too long if it is the command name, so we need to refer to the vocabulary to map the command name to an integer id as the item_id.
    dataset = nvt.Dataset(inter_data_path)
    workflow = nvt.Workflow(filtered_sessions)

    # The following will generate schema.pbtxt file in the provided folder and export the parquet files.
    workflow.fit_transform(dataset).to_parquet(data_path)

    workflow.save(workflow_path)


def save_random_split_balance_distribution(
    df,
    output_dir: str,
    train_size: float = 0.85,
    val_size: float = 0.15,
    overwrite: bool = True,
):
    
    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    exploded = df.explode('item_id-list')
    exploded = exploded[['session_id', 'item_id-list']].drop_duplicates()

    item_sessions = exploded.groupby('item_id-list')['session_id'].apply(
        lambda x: x.unique().tolist(),
        meta=('session_id', 'object')
    )


    item_sessions = item_sessions.compute()
    all_items = item_sessions.index

    random.seed(1)
    train_sessions = set()
    val_sessions   = set()

    for item_id, sessions in item_sessions.items():

        if len(sessions) < 2:

            print(f"Warning: item_id={item_id} only has {len(sessions)} sessions. "
                f"Cannot guarantee it appears in all splits!")
            for s in sessions:
                train_sessions.add(s)
                val_sessions.add(s)
            continue
        
        chosen_three = random.sample(sessions, 2)
        train_sessions.add(chosen_three[0])
        val_sessions.add(chosen_three[1])


    all_sessions = df['session_id'].unique().compute()
    all_sessions = set(all_sessions) 

    unassigned_sessions = list(all_sessions - train_sessions - val_sessions)

    random.shuffle(unassigned_sessions)

    train_ratio = train_size
    val_ratio   = val_size

    num_unassigned = len(unassigned_sessions)
    num_train = int(train_ratio * num_unassigned)
    num_val   = int(val_ratio * num_unassigned)

    train_sessions.update(unassigned_sessions[:num_train])
    val_sessions.update(unassigned_sessions[num_train:num_train+num_val])

    train_sessions_fs = frozenset(train_sessions)
    val_sessions_fs   = frozenset(val_sessions)

    train_df = df[df['session_id'].isin(train_sessions_fs)]
    val_df   = df[df['session_id'].isin(val_sessions_fs)]

    n_train = len(train_sessions)
    n_val   = len(val_sessions)

    n_total = len(all_sessions)

    print(f"Train sessions: {n_train} ({n_train / n_total * 100:.2f}%)")
    print(f"Val sessions:   {n_val}   ({n_val / n_total * 100:.2f}%)")

    schema = pa.schema([
    pa.field("session_id", pa.string()),
    pa.field("item_id-count", pa.int32()),
    pa.field("item_id-list", pa.list_(pa.int64())),
    pa.field("timestamp_interval-list", pa.list_(pa.float64())),
    pa.field("timestamp_interval_norm_global-list", pa.list_(pa.float64())),
    pa.field("merge_count-list", pa.list_(pa.float64())),
    pa.field("merge_count_norm-list", pa.list_(pa.float64())),
    pa.field("cat-list", pa.list_(pa.int64())),
    pa.field("classification-list", pa.list_(pa.int64())),
    pa.field("target-list", pa.list_(pa.int64())),
    pa.field("day_index", pa.int64()),
    pa.field("__null_dask_index__", pa.int64()),
    ])

    train_df = train_df.repartition(partition_size="600MB")
    train_df.to_parquet(os.path.join(output_dir, "train_new"),schema=schema, engine='pyarrow', write_metadata_file=True, overwrite=True)
    val_df = val_df.repartition(partition_size="600MB")
    val_df.to_parquet(os.path.join(output_dir, "val_new"), schema=schema, engine='pyarrow', write_metadata_file=True, overwrite=True)

    print("Done! Train/Val/ parquet files have been saved.")




if __name__ == "__main__":

    prepare_df_for_preprocessing_already_merged(path="data/1226_workflow_united_dropped_log_with_id.parquet", 
                                                output_path="data/interactions_merged_df_timestamp_feature_1226_latest.parquet")

    data = pd.read_parquet(os.path.join("data", "interactions_merged_df_timestamp_feature_1226_latest.parquet"))
    
    # after that we do augmentation to split long sessions to smaller ones
    data_augmented = split_sessions_with_correlated_ids(data, 
                                                        min_items=10,
                                                        max_items=100,
                                                        output_file="data/interactions_merged_df_timestamp_feature_augmented_1226_latest.parquet")

   # augment the data with side information
    data_augmented_llm = augment_data_with_llm_side_info(parquet_data="data/interactions_merged_df_timestamp_feature_augmented_1226_latest.parquet", 
                                                         llm_side_info_csv="data/combined_merged_message_counts_with_meanings_openai.csv",
                                                         output_file="data/interactions_merged_df_timestamp_feature_new_data_with_side_info_1226_latest.parquet")
    
    data_augmented = pd.read_parquet("data/interactions_merged_df_timestamp_feature_new_data_with_side_info_1226_latest.parquet", 
                                     columns=['session_id', 'item_id', 'timestamp', 'classification', 'target', 'timestamp_interval', 'merge_count', 'cat'])

    # Load the vocabulary mapping from 'item_id' strings to integer codes
    with open('data/1226voc_10workflows.json', 'r') as f:
        item_id_mapping = json.load(f)
    
    # Ensure the keys are strings and values are integers
    item_id_mapping = {str(k): int(v) for k, v in item_id_mapping.items()}
    
    # Map item_id strings to integer codes using the vocabulary mapping
    data_augmented['item_id'] = data_augmented['item_id'].map(item_id_mapping)

    # store the augmented data
    preprocess_data_path = "data/parquet_with_integer_id_for_preprocess_1226_latest.parquet"
    # if not os.path.exists(preprocess_data_path):
    data_augmented.to_parquet(preprocess_data_path, index=False)

    # let the nvtabular load the large data in chunks
    preprocessing(preprocess_data_path,
                data_path="data/processed_nvt_new_data_1226_latest",
                workflow_path="data/workflow_etl_new_data_1226_latest")
    
    # use dask to load the parquet files
    df = dd.read_parquet("data/processed_nvt_new_data_1226_latest/*.parquet")
    save_random_split_balance_distribution(df=df,
                                            output_dir="data/preproc_sessions_whole_new_data_1226_latest",
                                            train_size=0.85,
                                            val_size=0.15,
                                            overwrite=True)
