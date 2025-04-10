import asyncio
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import nvtabular
import pandas as pd
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from prototype.web_application.fastapi.fast_filtering import filtering
from web_application.fastapi.inference_request import test_triton_connection, triton_inference_web, extract_content

@asynccontextmanager
async def lifespan(app: FastAPI):
    global file_event_handler
    file_event_handler.loop = asyncio.get_running_loop()
    print("Lifespan startup: event loop updated.")
    yield

app = FastAPI(lifespan=lifespan)
messages_queue = asyncio.Queue()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, watch_path, grpc_endpoint, messages_queue, loop=None):
        super().__init__()
        self.watch_path = watch_path
        self.grpc_endpoint = grpc_endpoint
        self.messages_queue = messages_queue
        self.loop = loop  # this will be updated to the correct event loop in the startup event
        self.connections = []
        self.connection_status = test_triton_connection()
        self.workflow = nvtabular.Workflow.load(os.path.join("data", "workflow_etl_new_data_1226_latest"))
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

    def register(self, websocket: WebSocket):
        self.connections.append(websocket)

    def unregister(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    def on_modified(self, event):
        if event.src_path == self.watch_path:
            print("Log file updated!")
            try:
                df = pd.read_json(event.src_path, lines=True)
                filtered_df = filtering(df, self.lang_dict, self.cmd_mapping, self.vocabulary, self.augmentation_data)
                if self.connection_status:
                    if filtered_df is not None:
                        list_of_dicts = filtered_df.to_dict(orient='records')
                        commandHistory = [
                            {**row, 'message': self.id_name_mapping.get(row['item_id'], 'Unknown').replace("End Event: ", "")}
                            for row in list_of_dicts
                        ]
                        if filtered_df.shape[0] > 5:
                            prediction_json = triton_inference_web(self.workflow, filtered_df, self.id_mapping, self.id_name_mapping, endpoint=self.grpc_endpoint)
                            print("Inference done!")
                            prediction_json['commandHistory'] = commandHistory
                            if self.loop:
                                self.loop.call_soon_threadsafe(self.messages_queue.put_nowait, prediction_json)
                                print("Message sent to queue!")
                        else:
                            response_json = {
                                'item': ['No enough data for inference!'],
                                'item_scores': ['0'],
                                'commandHistory': commandHistory
                            }
                            if self.loop:
                                self.loop.call_soon_threadsafe(self.messages_queue.put_nowait, response_json)
                                print("History sent to queue!")
            except Exception as e:
                print(f"Error processing file modification: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    file_event_handler.register(websocket)
    try:
        while True:
            message = await messages_queue.get()
            await websocket.send_json(message)
            print("Message sent!")
    except WebSocketDisconnect:
        file_event_handler.unregister(websocket)
    finally:
        file_event_handler.unregister(websocket)
        print("WebSocket closed!")


def main():
    watch_path = "/mnt/c/Users/ge25yak/AppData/Roaming/Nemetschek/Vectorworks/2024/VW User Log.txt" 
    grpc_endpoint = "localhost:9999"  # SSH tunneling port
    global file_event_handler
    # initial loop is None, will be updated in the startup event
    file_event_handler = FileEventHandler(watch_path, grpc_endpoint, messages_queue, loop=None)
    observer = Observer()
    observer.schedule(file_event_handler, watch_path, recursive=False)
    observer.start()
    try:
        uvicorn.run(app, host="localhost", port=8000)
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()