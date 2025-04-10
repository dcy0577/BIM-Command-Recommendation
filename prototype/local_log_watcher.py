from datetime import datetime
import json
import os
import time
import pandas as pd
from prototype.web_application.fastapi.fast_filtering import filtering
# from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

class MyHandler(FileSystemEventHandler):
    def __init__(self, watch_path):
        super().__init__()
        self.watch_path = watch_path
        with open("data/1226voc_10workflows.json", "r") as f:
            self.vocabulary = json.load(f)
        self.augmentation_data = pd.read_csv("data/combined_merged_message_counts_with_meanings_openai.csv")
        self.lang_dict = pd.read_csv(os.path.join("data", "command_dictionary.csv"))
        self.cmd_mapping = pd.read_csv(os.path.join("data", "command_pairs_collections.csv"))
        df = pd.read_parquet(os.path.join("data", "categories", "unique.item_id.parquet")).reset_index().rename(columns={"index": "encoded_item_id"})
        # id_in_network -> item_id in vocabulary
        self.id_mapping = dict(zip(df['encoded_item_id'], df['item_id']))
        # item_id in vocabulary -> item_name
        self.id_name_mapping = {int(v): str(k) for k, v in self.vocabulary.items()}
        
    def on_modified(self, event):
        if event.src_path == self.watch_path:
            print("Log file updated!")
            # Read the file as a DataFrame
            df = pd.read_json(event.src_path, lines=True)
            filtered_df = filtering(df, self.lang_dict, self.cmd_mapping, self.vocabulary, self.augmentation_data)
            if filtered_df is not None:
                print(filtered_df.tail(10))
            else:
                print("filtered_df is None")

    def on_created(self, event):
        print("created")
    
    def on_deleted(self, event):
        print("deleted")
    
    def on_moved(self, event):
        print("moved")

def start_watching():
    path = "/mnt/c/Users/ge25yak/AppData/Roaming/Nemetschek/Vectorworks/2023/VW User Log.txt"
    event_handler = MyHandler(path)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watching()
