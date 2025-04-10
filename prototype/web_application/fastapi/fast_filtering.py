from datetime import datetime
import json
import os
import re
import pandas as pd
import numpy as np

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
    "(305)", #End Event: Edit Viewport (305) their appearance can be completely different from the original design layers, for presentation purposes.
    "(306)", #End Event: Exit Viewport (306)
    "(193)", # End Event: Set 3D View (193)
    "(8)",  # End Event: Change View (8)
    "(205)", # End Event: Fit to Page Area (205) The Fit to Objects command provides an easy way to zoom in and out of a drawing. There are two options: fit the window around all the objects in the drawing, or fit the window around a particular object or set of objects.
    "(203)", # End Event: Walkthrough (203) The Walkthrough tool simulates movement through a 3D model.
    
]

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

PREFIX_PATTERN = re.compile(r'^(?:' + '|'.join(PREFIXES_TO_REMOVE) + ')')
SUFFIX_PATTERN = re.compile(r'(?:' + '|'.join(re.escape(s) for s in SUFFIXES_TO_REMOVE) + r')$')

WORKFLOWS = [
    "Tool: Rectangle,End Event: Change Attributes",                   
    "Tool: Line,End Event: Delete,Tool: Line",
    "Tool: Line,Tool: Move by Points",
    "Tool: Rectangle,Menu: Add Surface - ",
    "Tool: Rectangle,Menu: Extrude and Edit - ",
    "Tool: 2D Polygon,End Event: Change Attributes",
    "Menu: Save As - ,Menu: Export PDF - ",
    "Menu: Duplicate - ,End Event: Drag,End Event: Modify Text",
    "Menu: Copy - ,End Event: Set Active Layer,Menu: Paste - ",
    "Tool: Mirror,End Event: Drag,End Event: Resize",
]

# Convert the timestamp in the ts column to UNIX timestamp
def convert_to_unix_timestamp(time_str):
    try:
        # split the time string into date, time, microseconds, and the rest
        date, time, microseconds, rest = time_str.split(None, 3)

        # combine date, time and timezone into a new time string
        new_time_str = f"{date} {time} {rest}"

        # convert the new time string to a datetime object
        dt = datetime.strptime(new_time_str, "%m/%d/%Y %H:%M:%S %z")

        # convert the microseconds to an integer before multiplying by 1000000
        microseconds = int(float(microseconds) * 1000000)

        # add the microseconds to the datetime object
        dt = dt.replace(microsecond=microseconds % 1000000)

        return int(dt.timestamp())
    except Exception as e:
        print(f"Error with timestamp: {time_str} - {e}")
        return 0

        
# Create a function to extract the command after the colon in the message
def extract_command(message):
    """# Create a function to extract the command after the colon in the message"""
    match = re.search(r': (.+)$', message)
    return match.group(1) if match else None


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

# Function to check if all characters in a string are ASCII
def is_ascii(s):
    s = str(s)
    return all(ord(c) < 128 for c in s)

#Combine with find_and_remove_single_matching_event
def remove_redo_undo(df):

    processed_data = []
    
    # Apply the find_and_remove_single_matching_event function and then drop marked rows
    recent_events = []
    recent_undone_events = []
    to_remove_indices = set()
    df.apply(find_and_remove_single_matching_event, axis=1, args=(recent_events, to_remove_indices, recent_undone_events))
    # Drop the marked rows
    group_df = df.drop(index=to_remove_indices)
    processed_data.append(group_df)

    # Combine all dataframes into one
    combined_dataframe = pd.concat(processed_data, ignore_index=True)
    return combined_dataframe

# Define a function that looks up the translation
def replace_with_translation(text, translation_dict):
    # Return the translated text if it exists in the dictionary, otherwise return the original text
    return translation_dict.get(text, text)

def multilingual_alignment(df, df_translation):
    # Create a dictionary from the DataFrame
    translation_dict = pd.Series(df_translation.label.values, index=df_translation.message).to_dict()
    # Replace messages in the 'message' column using the dictionary
    df['message'] = df['message'].str.replace("'", "", regex=False)

    df['message_eng'] = df['message'].apply(replace_with_translation, args=(translation_dict,))
            
    # New line added here to convert event message formats

    rows_to_drop = []
    # Iterate over DataFrame rows
    for index, row in df.iterrows():
        if not is_ascii(row['message_eng']):
            rows_to_drop.append(index)
    # Drop rows with non-ASCII characters in 'message_eng'
    df_to_update = df.drop(rows_to_drop)
    
    return df_to_update


def get_mapping_dict(tool_menu_dict, df_translation):
    for index, row in df_translation.iterrows():
        tool_menu_key = row['tool/menu']
        # If the key does not exist in the dictionary, create a new list
        if tool_menu_key not in tool_menu_dict:
            tool_menu_dict[tool_menu_key] = []
        # Append the 'Following_UNDO' to the list of the corresponding 'tool_menu'
        tool_menu_dict[tool_menu_key].append(
            row['event'] if pd.notnull(row['event']) else np.nan,
        )

def check_message(tool_menu_dict, tool_menu_key, undo_row) -> bool:
    if tool_menu_key not in tool_menu_dict:
        return False
    else:
        # Extract the 'message_eng' from the undo_row
        undo_message = undo_row['message_eng']
        # Check if the undo message is listed under the tool/menu key in the dictionary
        if undo_message in tool_menu_dict[tool_menu_key]:
            return True
        else:
            return False
            
def find_drop_rows(rows_to_drop, index, row, undo_rows, tool_menu_dict):
    
    self_triggered = False
    
    # If no matching 'UNDO' row is found, check if the tool_menu_key exists in the dictionary
    if row['message_eng'] not in tool_menu_dict:
        self_triggered = True
        
    else:
        # Iterate over the 'UNDO' rows
        for _, undo_row in undo_rows.iterrows():
            # Check if the 'UNDO' row's message matches the criteria for the 'Tool'/'Menu' event
            if check_message(tool_menu_dict, row['message_eng'], undo_row):
                # If a match is found, mark the 'UNDO' row for dropping
                rows_to_drop.append(undo_row.name)
                break  # No need to check further 'UNDO' rows


    # If no matching 'UNDO' row is found and the event is not self-triggered, consider dropping the event row
    if self_triggered is False and not any(check_message(tool_menu_dict, row['message_eng'], undo_row) for _, undo_row in undo_rows.iterrows()):
        rows_to_drop.append(index)

def replace_low_level(df, tool_menu_event_mapping):
    tool_menu_dict = {}
    get_mapping_dict(tool_menu_dict, tool_menu_event_mapping)  # Populate the tool_menu_dict
    rows_to_drop = []
    for index, row in reversed(list(df.iterrows())):
        if row['cat'] in ['Tool', 'Menu']:
            ts = row['ts']
            # Use boolean indexing to filter rows within the desired range
            sub_rows = df[(df['ts'] >= ts) & ~df.index.isin(rows_to_drop)]
            up_rows = df[(df['ts'] < ts) & ~df.index.isin(rows_to_drop)]

            # Find the first 'UNDO' action in these subsequent rows
            undo_rows = pd.concat([
                sub_rows[sub_rows['cat'] == 'UNDO'].head(5),
                up_rows[up_rows['cat'] == 'UNDO'].head(1)
            ])
            if not undo_rows.empty:
                find_drop_rows(rows_to_drop, index, row, undo_rows, tool_menu_dict)
            else:
                rows_to_drop.append(index)
    # Drop the rows that are to be filtered out
    df = df.drop(rows_to_drop).reset_index(drop=True)

    return df

def drop_certain_message(df):
    # Filter the DataFrame to exclude rows where 'message_eng' column exactly matches the specified string
    exclude_messages = [
    "End Event: Plug-in Event (166)", 
    "End Event:  (1)", 
    "End Event:  (166)", 
    "End Event: ()", 
    "End Event: Zoom on objects (166)", 
    "End Event: Zoom on objects (-1)",
    "Menu: Undo -  (-26) (0)",
    "Menu: Redo -  (-180) (0)"
    ]

    df = df[~df['message_eng'].isin(exclude_messages)]
    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)  # drop=True to avoid adding the old index as a column
    return df

def process_group(group_df):
    """
    Process each group to remove consecutive duplicate end events based on 'item_id'.
    Add a column to indicate the number of merging times for each row.
    """
    # Identify consecutive duplicates in 'message_content'
    is_duplicate = group_df['item_id'] == group_df['item_id'].shift(-1)

    # Initialize 'merge_count' column
    group_df['merge_count'] = 1

    # Iteratively process is_duplicate
    current_group_count = 1
    indices_to_keep = []  # Indices to keep (last one in each group)
    indices_to_update = []  # Indices where merge_count needs to be updated

    for idx, is_dup in enumerate(is_duplicate):
        if is_dup:  # If consecutive duplicate
            current_group_count += 1
        else:
            last_index = group_df.index[idx]  # Use the original group index
            indices_to_keep.append(last_index)
            indices_to_update.append((last_index, current_group_count))
            current_group_count = 1  # Reset the counter for the next group

    # Update merge_count for the last occurrence of each group
    for index, count in indices_to_update:
        group_df.at[index, 'merge_count'] = count

    # Filter DataFrame to keep only the marked indices
    result = group_df.loc[indices_to_keep].reset_index(drop=True)
    return result

def merge_workflows_in_group(group, workflow_patterns):
    indices_to_drop = []  
    merged_rows = []      
    item_ids = group['item_id'].tolist()
    n = len(item_ids)
    i = 0
    while i < n:
        matched = False

        for pattern in workflow_patterns:
            pat_len = len(pattern)

            if i + pat_len <= n:
                if item_ids[i:i+pat_len] == pattern:

                    new_row = group.iloc[i+pat_len-1].copy()

                    new_row['item_id'] = ", ".join(pattern) # there are spaces in between in the augmented data csv

                    new_row['cat'] = "workflow"
                    merged_rows.append(new_row)

                    indices_to_drop.extend(list(range(i, i+pat_len)))
                    i += pat_len
                    matched = True
                    break
        if not matched:
            i += 1

    group = group.drop(group.index[indices_to_drop])
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        group = pd.concat([group, merged_df], ignore_index=True)
        group = group.sort_values(by='timestamp').reset_index(drop=True)
    return group

def filtering(df, language_dict, commands_mapping, vocabulary, augmentation_data):
    # if df is not empty
    if df.empty:
        print("DataFrame is empty!") 
        return
    
    # Select only the required columns
    df = df[['ts', 'session', 'cat', 'message']]
    # filter the dataframe, only consider the rows where the column 'cat' contains the string 'UNDO'or 'TOOL' or 'MENU'
    df = df[df['cat'].str.contains('UNDO|Tool|Menu')]

    if df.empty:
        print("DataFrame is empty after filtering!")
        return
    
    # Remove rows with messages starting with the specified prefixes
    df_filtered_prefix = df[~df['message'].str.contains(PREFIX_PATTERN)]

    if df_filtered_prefix.empty:
        print("DataFrame is empty after filtering prefixes!")
        return
    
    ## Remove rows with messages ending with the specified suffixes (the command ids that we don't want to consider)
    df_filtered = df_filtered_prefix[~df_filtered_prefix['message'].str.contains(SUFFIX_PATTERN)]

    if df_filtered.empty:
        print("DataFrame is empty after filtering suffixes!")
        return
    
    df_filtered = df_filtered.reset_index()
    # Remove any text matching the pattern (MAX-<any number>)
    df_filtered['message'] = df_filtered['message'].str.replace(r'\(MAX-\d+\)', '', regex=True)

    ## Convert ts to a unified timestamp format
    df_filtered['ts'] = df_filtered['ts'].apply(convert_to_unix_timestamp)

    removed_redo_undo_df = remove_redo_undo(df_filtered)

    eng_df = multilingual_alignment(removed_redo_undo_df,language_dict)

    replaced_df = replace_low_level(eng_df,commands_mapping)

    final_data = drop_certain_message(replaced_df)

    pd.set_option('display.max_columns', None)
    print(final_data)

    if final_data.empty:
        print("processed_data is empty!")
        return
    
    # prepare input
    inter_data = final_data[['session', 'ts', 'message_eng','cat']].rename(columns={'ts': 'timestamp', 'message_eng': 'item_id', 'session': 'session_id'})

    # remove the suffixes in the item_id
    inter_data['item_id']= inter_data['item_id'].str.replace(r' (\(-?\d+\))+$', '', regex=True)

    # combine to the potential workflows
    workflow_patterns = [wf.split(",") for wf in WORKFLOWS]

    inter_data = inter_data.groupby('session_id', group_keys=False).apply(merge_workflows_in_group, workflow_patterns).reset_index(drop=True)

    # merge the consecutive commands
    inter_data = inter_data.groupby('session_id', group_keys=False).apply(process_group).reset_index(drop=True)

    # calculate the interval between timestamps within each session, create new column 'timestamp_interval'
    inter_data['timestamp_interval'] = inter_data.groupby('session_id')['timestamp'].diff().fillna(0)

    inter_data.dropna(inplace=True)

    if inter_data.empty:
        print("inter_data is empty!")
        return

    # load augmentation data
    inter_data = inter_data.merge(
        augmentation_data[['message_content', 'classification', 'target']],
        left_on='item_id',
        right_on='message_content',
        how='left'
        )

    inter_data.drop(columns='message_content', inplace=True)
    # drop the rows with NaN values in the 'classification' or 'target' column, as they are not included in the vocabulary as well
    inter_data.dropna(subset=['classification', 'target'], inplace=True)

    # convert name to id
    inter_data['item_id'] = inter_data['item_id'].apply(lambda x: vocabulary.get(x, -1))

    # reset the index
    inter_data.reset_index(drop=True, inplace=True)

    # inter_data.to_parquet(os.path.join("data", 'interactions_merged_df_timestamp_feature_new_data.parquet'), index=False)

    print("Parquet file saved!")
    print(inter_data.tail(5))

    return inter_data

if __name__ == "__main__":
    # Load the CSV with original and translated messages
    lang_dict = os.path.join("data","command_dictionary.csv")
    cmd_mapping = os.path.join("data","command_pairs_collections.csv")
    with open("data/1226voc_10workflows.json", "r") as f:
        vocabulary = json.load(f)
    augmentation_data = pd.read_csv("data/combined_merged_message_counts_with_meanings_openai.csv")
    df = pd.read_json("example_log_2024.txt", lines=True)
    filtered_df = filtering(df, lang_dict, cmd_mapping, vocabulary, augmentation_data)
    if filtered_df is not None:
        print(filtered_df.tail(10))
    else:
        print("filtered_df is None")