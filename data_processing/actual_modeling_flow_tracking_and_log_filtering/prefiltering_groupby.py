from datetime import datetime
import os
from tqdm import tqdm
import concurrent.futures
from functools import partial
import re
import pandas as pd
import pyarrow.parquet as pq

SUFFIXES_TO_REMOVE = [
    "(242)", # Event: zoom
    "(-242)",# Tool:Zoom
    "(218)", #End Event: Change View (218)
    "(199)", #End Event: Rotate 3D View (199)
    "(236)", # Event: pan
    "(-241)", # Tool: Pan
    "(-240)", # Tool : Select
    "(307)", # End Event: Force Select (307)
    "End Event: Select Similar: Undo not pos (166)",
    "End Event: Fit To Objects (166)",

    "(201)", # End Event: Flyover (201)
    "(-303)", # Tool: Flyover (-303)
    "(305)", #End Event: Edit Viewport (305) their appearance can be completely different from the original design layers, for presentation purposes.
    "(306)", #End Event: Exit Viewport (306)
    "(193)", # End Event: Set 3D View (193)
    "(8)",  # End Event: Change View (8)
    "(205)", # End Event: Fit to Page Area (205) The Fit to Objects command provides an easy way to zoom in and out of a drawing. There are two options: fit the window around all the objects in the drawing, or fit the window around a particular object or set of objects.
    "(203)", # End Event: Walkthrough (203) The Walkthrough tool simulates movement through a 3D model.
    
]

# to avoid removing the suffixes of the menu, in case "Menu: xxxx (333) (203)"
MENU_SUFFIXES_TO_REMOVE = []

PREFIXES_TO_REMOVE = [
    "DestroyEvent: ", 
    "Begin Internal Event: ", 
    "Event:", # lets focus on end event, but problem is end event also ends the internal event?
    "Event name changed from ", 
    "Beta Undo Alert", 
    "Undo Problem:",
    "Abort Event: ",
    "Menu: Undo",
    "Menu: Redo",
]

# Convert the timestamp in the ts column to UNIX timestamp
def convert_to_unix_timestamp(time_str):
    """Convert the timestamp in the ts column to UNIX timestamp"""
    try:
        if isinstance(time_str, pd.Timestamp):
            return int(time_str.timestamp())
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        return int(dt.timestamp())
    except Exception as e:
        print(f"Error with timestamp: {time_str} - {e}")
        return 0
        
# Create a function to extract the command after the colon in the message
def extract_command(message):
    """# Create a function to extract the command after the colon in the message"""
    match = re.search(r': (.+)$', message)
    return match.group(1) if match else None

# Create a function to extract the loc id of each commands
def extract_command_id(message):
    # Use regex to find patterns such as (-26) or (-5) (0)
    matches = re.findall(r'\((-?\d+)\)', message)

    results = ' '.join(matches)
    
    # If message starts with "Tool:", prepend 'T' to each matched ID
    if message.startswith("Tool:") and results:
        results = 'T: ' + results
    # If message starts with "Menu:", prepend 'M' to each matched ID
    elif message.startswith("Menu:") and results:
        results = 'M: ' + results
    # If message starts with "Event:", prepend 'E' to each matched ID
    elif message.startswith("End Event:") and results:
        results = 'E: ' + results
    return results

#find certain rows, which contains redo and undo, and it's relative end event
def find_and_remove_single_matching_event(row, recent_events, to_remove_indices, recent_undone_events):
    idx = row.name
    message = row['message']

    # remove the event from remove list if it is redone by redo event
    if message.startswith("Redo Event:"):
        command = extract_command(message).strip()
        # Check if the command matches any of the recent events
        matching_event_index = next((index for index, evt in reversed(recent_undone_events) if evt == command), None)
        # If a matching event is found, this evnet is redone, remove it from the list of undone events as well as the list of events to remove
        # Also, add the current row from the list of events to remove
        if matching_event_index is not None:
            to_remove_indices.remove(matching_event_index)
            to_remove_indices.add(idx)
            recent_undone_events.remove((matching_event_index, command))
        else:
            # If no matching event is found, just mark the current row for removal
            to_remove_indices.add(idx)
        
    # Check if message starts with "Undo and Remove Action: " or "Abort Event: " or "Undo Event: "
    if message.startswith("Undo and Remove Action: ") or message.startswith("Undo Event: "):
        command = extract_command(message).strip()
        
        # Check if the command matches any of the recent events
        matching_event_index = next((index for index, evt in reversed(recent_events) if evt == command), None)
        
        # If a matching event is found, mark both the current and the matching rows for removal
        # Also, remove the matching event from the list of recent events
        if matching_event_index is not None:
            to_remove_indices.add(matching_event_index)
            to_remove_indices.add(idx)
            recent_events.remove((matching_event_index, command))
            # record the undone event
            if message.startswith("Undo Event: "):
                recent_undone_events.append((matching_event_index, command))
        else:
            # If no matching event is found, just mark the current row for removal
            to_remove_indices.add(idx)

    # If the message starts with "Event:", add its command and its index to the list of recent events
    if message.startswith("End Event:"):
        recent_events.append((idx, extract_command(message).strip()))


    return row


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

def get_all_files_from_directory_list(directory_list):
    all_files = []
    for directory_path in directory_list:
        all_files.extend(get_all_files(directory_path))
    return all_files

def process_file(full_path, prefix_pattern, suffix_pattern_menu, suffix_pattern_others):
    try:
        # Load the parquet file
        # iter_df = pd.read_parquet(full_path, 
        #                         engine='pyarrow', 
        #                         columns=['session_anonymized', 'ts', 'cat', 'message'],
        #                         chunksize=10000)
        
        parquet_file = pq.ParquetFile(full_path)
        filtered_chunks = []

        for i in range(parquet_file.num_row_groups):
            # Read a single row group at a time
            table = parquet_file.read_row_group(i, columns=['session_anonymized', 'ts', 'cat', 'message'])
            df_chunk = table.to_pandas()

            # retain rows only with the cat column values of UNDO, Menu, Tool
            df_filtered = df_chunk[df_chunk['cat'].isin(['UNDO', 'Menu', 'Tool'])]
            # Apply regex filters directly, avoiding creating additional DataFrames
            df_filtered = df_filtered[~df_filtered['message'].str.contains(prefix_pattern, regex=True)]
            df_filtered = df_filtered[~df_filtered['message'].str.contains(suffix_pattern_menu, regex=True)]
            df_filtered = df_filtered[~df_filtered['message'].str.contains(suffix_pattern_others, regex=True)]
            # In-place modifications to reduce memory 
            df_filtered['message']= df_filtered['message'].replace(r'\(MAX-\d+\)', '', regex=True)

            df_filtered['ts'] = df_filtered['ts'].apply(convert_to_unix_timestamp)
            filtered_chunks.append(df_filtered)

        return pd.concat(filtered_chunks, ignore_index=True)

    except Exception as e:
        raise Exception(f"Error processing file {full_path}: {e}")

#Combine all the files
def combine_remove_filter(directory_path):
    # rank subdirectories by the name, and list all files in the all subdirectories

    parquet_files_abs_paths = get_all_files(directory_path)

    # precompile the regex pattern for efficiency
    prefix_pattern = re.compile(r'^(?:{})'.format('|'.join(PREFIXES_TO_REMOVE)))
    if MENU_SUFFIXES_TO_REMOVE:
        suffix_pattern_menu = re.compile(r'Menu: .*' + r'(?:{})$'.format('|'.join([re.escape(suffix) for suffix in MENU_SUFFIXES_TO_REMOVE])))
    else:
        # will never match
        suffix_pattern_menu = re.compile(r'a^')
    suffix_pattern_others = re.compile(r'^(?!Menu:).*' + r'(?:{})$'.format('|'.join([re.escape(suffix) for suffix in SUFFIXES_TO_REMOVE])))

    # Create a new function with some arguments already filled in
    partial_process_file = partial(process_file, 
                                   prefix_pattern=prefix_pattern, 
                                   suffix_pattern_menu=suffix_pattern_menu, 
                                   suffix_pattern_others=suffix_pattern_others)

    list_of_dataframes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for df in tqdm(executor.map(partial_process_file, parquet_files_abs_paths), 
                       total=len(parquet_files_abs_paths), 
                       desc="Processing parquet files"):
            list_of_dataframes.append(df)
        
    combined_dataframe = pd.concat(list_of_dataframes, ignore_index=True)
    return combined_dataframe

def process_group(group_df):
    # oder the data within group by timestamp
    group_df.sort_values(by='ts', inplace=True)
    # Apply the find_and_remove_single_matching_event function and then drop marked rows
    recent_events = []
    recent_undone_events = []
    to_remove_indices = set()
    group_df.apply(find_and_remove_single_matching_event, axis=1, args=(recent_events, to_remove_indices, recent_undone_events))
    # Drop the marked rows
    group_df.drop(index=to_remove_indices, inplace=True)
    return group_df

#Combine with find_and_remove_single_matching_event
def remove_redo_undo(df):

    processed_data = []
    grouped_data = df.groupby('session_anonymized')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for processed_group in tqdm(executor.map(process_group, [group for _, group in grouped_data]), 
                                    total=len(grouped_data), 
                                    desc="Processing grouped data for undo redo logics"):
            processed_data.append(processed_group)

    # Combine all dataframes into one
    return pd.concat(processed_data, ignore_index=True)

def generate_partial_groupby_files(parquet_path='data/all_data_after_redo_undo.parquet'):
    # default reading, whole file is loaded into memory, ~90GB
    df = pd.read_parquet(parquet_path)
    # groupby session_anonymized and sort by ts, concatenate every 10000 groups into one dataframe and save it to parquet
    print("loaded data!")
    grouped = df.groupby('session_anonymized')
    saved_groupes = []
    for i, (_, group) in tqdm(enumerate(grouped), desc="Grouping data", total=len(grouped)):
        saved_groupes.append(group)
        if i % 10000 == 0:
            print(f"Saving groups {i}")
            pd.concat(saved_groupes, ignore_index=True).to_parquet(f'data/groupby/grouped_data_{i}.parquet')
            # clear the list
            saved_groupes = []
    # save the rest
    pd.concat(saved_groupes, ignore_index=True).to_parquet(f'data/groupby/grouped_data_{i}.parquet')

if __name__ == "__main__":
    # this part tooks 20 hours on pointy with 5 workers (not to fill the memory)
    combined_df = combine_remove_filter("/home/cdu/vw_log_data") # prefiltering the raw log data
    removed_redo_undo_df = remove_redo_undo(combined_df)
    removed_redo_undo_df.to_parquet('data/actual_modeling_Logs.parquet')
    # we also save the data into smaller chunks to reduce the RAM usage for downstraem processing
    generate_partial_groupby_files('data/actual_modeling_Logs.parquet')

