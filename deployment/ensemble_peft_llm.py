from itertools import accumulate
import os
import shutil
import sys
import numpy as np
import nvtabular
import torch
from merlin.systems.dag import Ensemble
from merlin.systems.dag.ops.pytorch import PredictPyTorch
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.dataloader.ops.embeddings import EmbeddingOperator
import cloudpickle
# need make sure this is our modified peft
from peft import PeftModel
sys.path.insert(0, 'transformers4rec')
from model.train_eval_full_models import model


def _get_values_offsets(data):
    values = []
    row_lengths = []
    for row in data:
        row_lengths.append(len(row))
        values += row
    offsets = [0] + list(accumulate(row_lengths))
    if all(isinstance(i, torch.Tensor) and i.dim()==0 for i in values):
        return torch.tensor(values), torch.tensor(offsets)
    else:
        return torch.cat(values), torch.tensor(offsets)

def generate_input_dict(dataloader):
    model_input_dict_from_dataloaer = next(iter(dataloader))[0]
    model_input_dict = {}
    item_id_list__values, item_id_list__offsets = _get_values_offsets(model_input_dict_from_dataloaer["item_id-list"])
    model_input_dict["item_id-list__values"] = item_id_list__values.to("cuda")
    model_input_dict["item_id-list__offsets"] = item_id_list__offsets.to("cuda")
    timestamp_interval_norm_global_list__values, timestamp_interval_norm_global_list__offsets = _get_values_offsets(model_input_dict_from_dataloaer["timestamp_interval_norm_global-list"])
    model_input_dict["timestamp_interval_norm_global-list__values"] = timestamp_interval_norm_global_list__values.to("cuda")
    model_input_dict["timestamp_interval_norm_global-list__offsets"] = timestamp_interval_norm_global_list__offsets.to("cuda")
    pretrained_item_id_embeddings__values, pretrained_item_id_embeddings__offsets = _get_values_offsets(model_input_dict_from_dataloaer["pretrained_item_id_embeddings"])
    model_input_dict["pretrained_item_id_embeddings__values"] = pretrained_item_id_embeddings__values.reshape(-1, 3072).to("cuda") # 3072 is the dim of pretrained embeddings
    model_input_dict["pretrained_item_id_embeddings__offsets"] = pretrained_item_id_embeddings__offsets.to("cuda")

    classification_list__values, classification_list__offsets = _get_values_offsets(model_input_dict_from_dataloaer["classification-list"])
    model_input_dict["classification-list__values"] = classification_list__values.to("cuda")
    model_input_dict["classification-list__offsets"] = classification_list__offsets.to("cuda")

    target_list__values,target_list__offsets = _get_values_offsets(model_input_dict_from_dataloaer["target-list"])
    model_input_dict["target-list__values"] = target_list__values.to("cuda")
    model_input_dict["target-list__offsets"] = target_list__offsets.to("cuda")

    merge_count_norm__values,merge_count_norm__offsets = _get_values_offsets(model_input_dict_from_dataloaer["merge_count_norm-list"])
    model_input_dict["merge_count_norm-list__values"] = merge_count_norm__values.to("cuda")
    model_input_dict["merge_count_norm-list__offsets"] = merge_count_norm__offsets.to("cuda")

    return model_input_dict


def export_ensemble_peft(embeddings_op, 
                         adapter_checkpoint_path = "tmp_clm_llama/checkpoint-800/adapter_model",
                         model_name = "llama_lora",
                         workflow_path = "data/workflow_etl_new_data_1226_latest",
                         ens_model_path = "ens_models_peft"):

    # TODO: check whether need to remove prepare_model_for_kbit_training(model) in to_huggingface_torch_model()
    pretrained_model, schema, max_sequence_length, _, val_data_loader = model(dataset_path="data/processed_nvt_new_data_1226_latest", 
                                            model_name = model_name,
                                            val_batch_size=2,
                                            val_paths="sub_trainset_for_att_visualization.parquet",)
    
    pretrained_model.eval() 
    # this function is actually loading adapter weights to the tr_model, so the peft_model is actually the tr_model
    peft_model = PeftModel.from_pretrained(
            pretrained_model,
            model_id = adapter_checkpoint_path,
            torch_dtype=torch.float16,
            device_map= 0, # auto is not good for fine-tuning, load the model to gpu
        )
        
    print(peft_model)
    tr_model = peft_model.merge_and_unload(progressbar=True)
    print(tr_model)
    tr_model.eval()
    tr_model.to("cuda:0")

    # changes output schema from "next items" to "items scores" and "items ids" 
    topk = 5
    tr_model.top_k = topk

    tr_model.to("cuda:0")
    tr_model.eval()
    print(tr_model.input_schema)
    print(tr_model.output_schema)

    # TODO: trace model does not work for peft model
    # model_input_dict = generate_input_dict(dataloader)
    # traced_model = torch.jit.trace(tr_model, model_input_dict, strict=False)

    # use cloudpickle to store the weights and architecture, and serve it on triton using customized python backend
    with open("model.pkl", "wb") as f:
        cloudpickle.dump(tr_model, f)

    # overwrite the existing ens_model_path
    if os.path.exists(ens_model_path):
        shutil.rmtree(ens_model_path)  
    os.mkdir(ens_model_path) 

    workflow = nvtabular.Workflow.load(workflow_path)

    torch_op = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> embeddings_op >> PredictPyTorch(
    tr_model, tr_model.input_schema, tr_model.output_schema)

    ensemble = Ensemble(torch_op, workflow.input_schema)
    ens_config, node_configs = ensemble.export(ens_model_path)
    print(ens_config)
    print(node_configs)


def export_ensemble_standard(embeddings_op, 
                             checkpoint_path = "mlm_mixtral_2_layers_all_stuff/checkpoint-520000/pytorch_model.bin", 
                             workflow_path = "data/workflow_etl_new_data_1226_latest", 
                             ens_model_path = "ens_models_mixtral_new",
                             model_name="mixtral"):

    tr_model, schema, max_sequence_length, _, val_data_loader = model(dataset_path="data/processed_nvt_new_data_1226_latest", 
                                                    model_name = model_name,
                                                    val_batch_size=2,
                                                    val_paths="sub_trainset_for_att_visualization.parquet",)
    
    # Restoring model weights
    tr_model.load_state_dict(torch.load(checkpoint_path))
    # Restoring random state
    # rng_file = os.path.join(checkpoint_path, "rng_state.pth")
    # checkpoint_rng_state = torch.load(rng_file)
    # random.setstate(checkpoint_rng_state["python"])
    # np.random.set_state(checkpoint_rng_state["numpy"])
    # torch.random.set_rng_state(checkpoint_rng_state["cpu"])
    # torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])

    # changes output schema from "next items" to "items scores" and "items ids" 
    topk = 5
    tr_model.top_k = topk

    tr_model.to("cuda")
    tr_model.eval()
    print(tr_model.input_schema)
    print(tr_model.output_schema)

    # TODO trace model does not work, so we use cloudpickle and customized python backend on triton
    # model_input_dict = generate_input_dict(val_data_loader)
    # traced_model = torch.jit.trace(tr_model, model_input_dict, strict=False)

    with open("model.pkl", "wb") as f:
        cloudpickle.dump(tr_model, f)

    # overwrite the existing ens_model_path
    if os.path.exists(ens_model_path):
        shutil.rmtree(ens_model_path)  
    os.mkdir(ens_model_path) 

    workflow = nvtabular.Workflow.load(workflow_path)

    torch_op = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> embeddings_op >> PredictPyTorch(
    tr_model, tr_model.input_schema, tr_model.output_schema)

    ensemble = Ensemble(torch_op, workflow.input_schema)
    ens_config, node_configs = ensemble.export(ens_model_path)
    print(ens_config)
    print(node_configs)

if __name__ == "__main__":

    pretrained_emb_path = "data/pre-trained-item-id-new-data_0122.npy"
    np_emb_item_id = np.load(pretrained_emb_path) # (item_cardinality, pretrained_dim)
    embeddings_op = EmbeddingOperator(
        np_emb_item_id, lookup_key="item_id-list", embedding_name="pretrained_item_id_embeddings"
    )

    export_ensemble_standard(embeddings_op, model_name="mixtral")
