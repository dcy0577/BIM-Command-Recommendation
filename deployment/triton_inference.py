import json
import os
import nvtabular
import pandas as pd
import tritonclient.http as client
from merlin.systems.triton.utils import send_triton_request

def map_encoded_item_id_to_original_item_id(response):
    df =  pd.read_parquet('categories/unique.item_id.parquet').reset_index().rename(columns={"index": "encoded_item_id"})
    id_mapping = dict(zip(df['encoded_item_id'], df['item_id']))
    # Load the vocabulary mapping from 'item_id' strings to integer codes
    with open('data/1226voc_10workflows.json', 'r') as f:
        item_id_mapping = json.load(f)
    # Ensure the values are strings and keys are integers
    id_to_name_mapping = {int(v): str(k) for k, v in item_id_mapping.items()}

    for i in range(len(response['item_ids'])):
        item_ids = response['item_ids'][i]
        item_id_scores = response['item_id_scores'][i]
    
        # map item_ids
        mapped_item_ids = [id_mapping.get(item_id, 'Unknown') for item_id in item_ids]

        # mapped id to name
        mapped_item_names = [id_to_name_mapping.get(mapped_item_id, 'Unknown') for mapped_item_id in mapped_item_ids]

        # output string
        output_str = "Top 5 itemIDs and name for next item are: "
        for i, (item_id, item_name, score) in enumerate(zip(mapped_item_ids, mapped_item_names,item_id_scores)):
            output_str += f"{item_id} ({item_name}) (score: {score})"
            if i < len(mapped_item_ids) - 1:
                output_str += ", "

        print(f"for this session, {output_str}")

def triton_inference():
    # docker run --gpus=all --shm-size=1G -p8000:8000 -p8001:8001 -p8002:8002 --rm -v ./ens_models:/models -v ./t4rec_23.06.tar.gz:/t4rec_23.06.tar.gz --name tritonserver nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-repository=/models
    # tritonserver --model-repository=/ens_models

    workflow = nvtabular.Workflow.load(os.path.join("data", "workflow_etl_new_data_1226_latest"))

    try:
        triton_client = client.InferenceServerClient(url="localhost:8000", verbose=True)
        print("client created.")
    except Exception as e:
        print("channel creation failed: " + str(e))
    triton_client.is_server_live()
    triton_client.get_model_repository_index()

    # this should be the output of the filtering script
    interactions_merged_df = pd.read_parquet(os.path.join("data", "sub_testset_for_triton_inference.parquet"))

    batch = interactions_merged_df[110:122]
    # sessions_to_use = batch.session_id.value_counts()
    # filtered_batch = batch[batch.session_id.isin(sessions_to_use[sessions_to_use.values>1].index.values)]
    print("==========original================")
    print(batch)

    input_batch = batch[:-1]
    print("==========input_batch================")
    print(input_batch)
    print("==========label================")
    label_batch = batch[-1:]
    print(label_batch)
    print("==========response================")

    # ['item_id_scores', 'item_ids'] is the tr_model.output_schema.column_names
    response = send_triton_request(workflow.input_schema, input_batch, ['item_id_scores', 'item_ids'])
    print(response)
    map_encoded_item_id_to_original_item_id(response)


if __name__ == "__main__":
    triton_inference()