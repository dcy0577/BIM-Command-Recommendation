import json
import os
import os
import nvtabular
import pandas as pd
import tritonclient.grpc as grpcclient
from merlin.systems.triton.utils import send_triton_request
from dotenv import load_dotenv
load_dotenv()
endpoint = os.getenv("TRITON_GRPC_ENDPOINT")


def extract_content(item_names):
    return [item_name.replace("End Event: ", "") for item_name in item_names]

def test_triton_connection():
    try:
        # triton_client = client.InferenceServerClient(url=http_endpoint, verbose=True)
        triton_client = grpcclient.InferenceServerClient(url="localhost:9999", verbose=True)
        print("client created.")
    except Exception as e:
        print("channel creation failed: " + str(e))
    try:
        is_live = triton_client.is_server_live()
        triton_client.get_model_repository_index()
        return is_live
    except Exception as e:
        print(str(e))


def triton_inference_web(workflow, input_batch, id_mapping, id_name_mapping, endpoint):
    
    response = send_triton_request(workflow.input_schema, input_batch, ['item_id_scores', 'item_ids'], endpoint=endpoint)
    print(response)

    if len(response['item_ids']) > 1:
        raise NotImplementedError("session amount > 1 is not supported yet!")
    
    # create an emtpy json to store the item and item_scores
    json_response = {}

    # in this prototype, we usually only have one session, we take the last one
    item_ids = response['item_ids'][-1]
    item_id_scores = response['item_id_scores'][-1]

    # map item_ids
    mapped_item_ids = [id_mapping.get(item_id, 'Unknown') for item_id in item_ids]

    # get the item names
    item_names = [id_name_mapping.get(int(item_id), 'Unknown') for item_id in mapped_item_ids]

    # extract content from the mapped_item_ids
    extract_content_mapped_items = extract_content(item_names)

    json_response['item'] = extract_content_mapped_items
    json_response['item_scores'] = [str(score) for score in item_id_scores]

    print(json_response)

    return json_response


def test_triton_web():
    is_live = test_triton_connection()
    if is_live:
        workflow = nvtabular.Workflow.load(os.path.join("data", "workflow_etl_new_data_1226_latest"))
        batch = pd.read_parquet(os.path.join("data", "sub_testset_for_triton_inference.parquet"))
        with open("data/1226voc_10workflows.json", "r") as f:
            vocabulary = json.load(f)
        df =  pd.read_parquet(os.path.join("data", "categories","unique.item_id.parquet")).reset_index().rename(columns={"index": "encoded_item_id"})
        # id_in_network -> item_id in vocabulary
        id_mapping = dict(zip(df['encoded_item_id'], df['item_id']))
        # item_id in vocabulary -> item_name
        id_name_mapping = {int(v): str(k) for k, v in vocabulary.items()}

        batch= batch[2:23]

        print("==========original================")
        print(batch)

        input_batch = batch[:-1]
        print("==========input_batch================")
        print(input_batch)
        print("==========label================")
        label_batch = batch[-1:]
        print(label_batch)
        print("==========response================")

        triton_inference_web(workflow, input_batch, id_mapping, id_name_mapping, endpoint="localhost:9999")

if __name__ == "__main__":
    test_triton_web()