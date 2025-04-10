import json
import os
import pandas as pd
import torch
from tqdm import tqdm
import concurrent.futures
import pyarrow.parquet as pq
from functools import partial
import concurrent.futures
from collections import Counter, defaultdict
from concurrent.futures import as_completed


def get_sub_df(df_path, rows, output_path):
    df = pd.read_parquet(df_path)
    new_df = df.iloc[rows[0]:rows[1]]
    new_df.to_parquet(output_path)
    pd.set_option('display.max_columns', None)
    print(new_df.tail(20))


def from_vocab_id_to_name_to_IdInNetwork(vocabulary='data/vocabulary_12000.json', 
                                         llm_side_info_csv='data/combined_merged_message_counts_with_meanings_openai.csv',
                                         output_path="unique.item_id_name_and_size.parquet"):
    df =  pd.read_parquet('categories/unique.item_id.parquet')
    # the item_id is the vocabulary id
    vocal_to_IdInNetwork = dict(zip(df["item_id"], df.index))
    df["id_in_network"] = df["item_id"].map(vocal_to_IdInNetwork)
    # now vocal id to name
    with open(vocabulary, 'r') as f:
        item_id_mapping = json.load(f)
    # reverse the dictionary
    item_id_mapping = {int(v): str(k) for k, v in item_id_mapping.items()}
    
    # Map 'item_id' strings to integer codes using the vocabulary mapping
    df['item_id_name'] = df['item_id'].map(item_id_mapping)

    df_llm_csv = pd.read_csv(llm_side_info_csv)

    df_merged = df.merge(
    df_llm_csv[['message_content', 'classification', 'target']],
    left_on='item_id_name',
    right_on='message_content',
    how='left'
    )

    df_merged.drop(columns='message_content', inplace=True)
    
    df_merged.to_parquet(output_path)


def get_row_info_from_session_id():
    df = pd.read_parquet("data/interactions_merged_df_timestamp_feature_augmented_new_data.parquet")
    df = df[df['session_id'] == "00189F63-1"]
    pd.set_option('display.max_columns', None)
    print(df)

def compute_class_weights(parquet_path="categories/unique.item_id.parquet"):
    df = pd.read_parquet(parquet_path)
    # get the fist column name in string
    first_column_name = df.columns[0]
    size_name = first_column_name + "_size"
    total_samples = df[size_name].sum()
    num_classes = df.shape[0]+3
    df['class_weight'] = total_samples / (num_classes * df[size_name])
    df['class_weight_normalized'] = df['class_weight'] / df['class_weight'].sum()
    weights = torch.tensor(df['class_weight_normalized'].values, dtype=torch.float)
    new_tensor = torch.tensor([1e-8, 1e-8, 1e-8], dtype=torch.float)
    final_weights = torch.cat((new_tensor, weights))
    return final_weights

# for counting the command frequency in prefiltered data after tracking the redo/undo
def command_freq(path):
    # default reading, whole file is loaded into memory, ~90GB
    df = pd.read_parquet(path, columns=['message_content'])
    message_counts = df.groupby('message_content').size()
    result_df = pd.DataFrame(message_counts, columns=['count'])
    # order by the counts
    result_df.sort_values(by='count', ascending=False, inplace=True)
    result_df.to_parquet(path + '_message_counts.parquet')

def find_missing_row(parquet_path):
    df_merged = pd.read_parquet(parquet_path, columns=['session_id', 'item_id', 'classification', 'target'])
    missing_rows = df_merged[df_merged['classification'].isna() | df_merged['target'].isna()]
    missing_rows.to_parquet("data/missing_rows.parquet")

def get_sessions_has_minority_classes(parquet_path, minority_output_path):
    df = pd.read_parquet(parquet_path)
    # lets getlow frequency items
    df_full_item_id = pd.read_parquet("unique.item_id_name_and_size.parquet")
    item_frequency_threshold = 10
    df_full_item_id = df_full_item_id[df_full_item_id['item_id_size'] < item_frequency_threshold]

    item_names = df_full_item_id['item_id_name'].unique()
    matching_sessions = df[df['item_id'].isin(item_names)]['session_id'].unique()
    filtered_df_has_minority_classes = df[df['session_id'].isin(matching_sessions)]
    
    filtered_df_has_minority_classes.to_parquet(minority_output_path)
    
    num_sessions = df['session_id'].nunique()
    num_sessions_with_minority_classes = filtered_df_has_minority_classes['session_id'].nunique()
    print(f"original splitted dataset has {num_sessions} sessions, among them there are {num_sessions_with_minority_classes} sessions that have minority classes")


def check_split_data_minority_classes(train_path, val_path, minority_output_path):
    df = pd.read_parquet(minority_output_path)
    sessions = df['session_id'].unique()
    # check in training set
    train_df = pd.read_parquet(train_path)
    train_matching_sessions_n = train_df[train_df['session_id'].isin(sessions)]['session_id'].nunique()
    # check in validation set
    val_df = pd.read_parquet(val_path)
    val_matching_sessions_n = val_df[val_df['session_id'].isin(sessions)]['session_id'].nunique()

    print(f"train dataset has {train_matching_sessions_n} sessions that have minority classes, validation dataset has {val_matching_sessions_n} sessions that have minority classes")

def convert_id_to_name():
    df = pd.read_parquet("sub_trainset_for_att_visualization.parquet")
    df_map = pd.read_parquet("unique.item_id_name_and_size.parquet")
    mapping_dict = df_map.set_index('id_in_network')['item_id_name'].to_dict()
    def map_ids(id_list):
        return [mapping_dict.get(i, f"not found({i})") for i in id_list]
    df['item_id_name_list'] = df['item_id-list'].apply(map_ids)
    df = df[['session_id', 'item_id_name_list', 'item_id-list']]
    df.to_parquet("sub_trainset_for_att_visualization_with_name.parquet")

# for counting the command frequency in raw data
def process_raw_file_count_messages(full_path):
    try:
        parquet_file = pq.ParquetFile(full_path)
        message_counts = defaultdict(int)

        for i in range(parquet_file.num_row_groups):
            # Read a single row group at a time
            table = parquet_file.read_row_group(i, columns=['message'])
            df_chunk = table.to_pandas()
            # In-place modifications to reduce memory 
            df_chunk['message']= df_chunk['message'].replace(r'\(MAX-\d+\)', '', regex=True)
            
            # Count occurrences of each message in this chunk
            chunk_counts = df_chunk['message'].value_counts().to_dict()
            
            # Aggregate counts to the total message_counts
            for message, count in chunk_counts.items():
                message_counts[message] += count
            del df_chunk  

        return message_counts

    except Exception as e:
        raise Exception(f"Error processing file {full_path}: {e}")

# for counting the command frequency in raw data
def merge_counts(counts_list):
    total_counts = defaultdict(int)
    for counts in counts_list:
        for message, count in counts.items():
            total_counts[message] += count
    return total_counts

# for counting the command frequency in raw data
def raw_data_command_freq():
    parquet_files_abs_paths = get_all_files("/home/cdu/vw_log_data")
    # Create a new function with some arguments already filled in
    partial_process_file = partial(process_raw_file_count_messages)

    counts_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for counts in tqdm(executor.map(partial_process_file, parquet_files_abs_paths), 
                           total=len(parquet_files_abs_paths), 
                           desc="Processing parquet files"):
            counts_list.append(counts)

    print("Finished processing all files, start combining message counts")

    # Merge all counts
    total_counts = merge_counts(counts_list)
    print("Finished combining message counts, start saving to parquet")

    # Convert the dictionary to a DataFrame
    result_df = pd.DataFrame(list(total_counts.items()), columns=['message', 'count'])
    # order by the counts
    result_df.sort_values(by='count', ascending=False, inplace=True)
    result_df.to_parquet('data/raw_data_message_counts.parquet', index=False)


def get_all_files(directory_path):
    all_files = []
    # Get all subdirectories and files
    for root, dirs, files in os.walk(directory_path):
        # Sort directories and files
        dirs.sort()
        files.sort()
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

# for calculating the average session length in raw data
def process_raw_file(file_path):
        counter = Counter()
        try:
            pf = pq.ParquetFile(file_path)

            for rg in range(pf.num_row_groups):

                table = pf.read_row_group(rg, columns=["session_anonymized"])

                session_ids = table.column("session_anonymized").to_pylist()
                counter.update(session_ids)
        except Exception as e:
            print(f"processing {file_path} triggered error: {e}")
        return counter

# for calculating the average session length in raw data
def raw_session_avg_length_count():

    file_list = get_all_files("/home/cdu/vw_log_data")
    global_counter = Counter()

    num_workers = 10
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:

        future_to_file = {executor.submit(process_raw_file, fp): fp for fp in file_list}

        for future in tqdm(as_completed(future_to_file), total=len(file_list), desc="processing files"):
            file_path = future_to_file[future]
            try:
                file_counter = future.result()

                global_counter.update(file_counter)
            except Exception as e:
                print(f"processing {file_path} triggered error: {e}")
                
    total_sessions = len(global_counter)
    total_items = sum(global_counter.values())
    avg_session_length = total_items / total_sessions if total_sessions > 0 else 0

    print("\nResult：")
    print(f"  total session count: {total_sessions}")
    print(f"  total item count: {total_items}")
    print(f"  avg session length: {avg_session_length:.2f}")


if __name__ == '__main__':
    get_sub_df("data/preproc_sessions_whole_new_data_1226_latest/val_new/part.0.parquet", (0, 64), "sub_trainset_for_att_visualization.parquet")