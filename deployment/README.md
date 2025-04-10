#  Ensemble model for deployment
Run `deployment/ensemble_peft_llm.py` to generate deployable workflow on the triton server. The script will generate a folder structure like
```
ens_models_mixtral
  L 0_transformworkflowtriton
    L 1
    L config.pbtxt
  L 1_predictpytorchtriton
    L 1
    L config.pbtxt
  L executor_model
    L 1
    L config.pbtxt
```
and a cloud pickle file `model.pkl` with weights and architecture.

# Pack the conda env using conda-pack
We need to mount the packed conda env in docker, so that the triton server python backend sub can use the env to excute workflow in model.py.
To generate t4rec_23.06_new_transformers.tar.gz:
```
conda activate t4rec_23.06_new_transformers
conda-pack
```

# Change config.pbtxt in the ens_models folder
1. Add in each config.phtxt:
```
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "/t4rec_23.06.tar.gz"}
}
```
reference: https://github.com/triton-inference-server/python_backend/#creating-custom-execution-environments

2. In `1_predictpytorchtriton/1/`
    0. Remove the `model.pt` folder
    1. Copy the saved `mode.pkl` to here
    2. Copy `model.py` from `deploy_template` to here
    3. Replace `config.pbtxt` with `deploy_template/config.pbtxt`. Specifically, following changes are made in the template: 
    ```
    input {
      name: "pretrained_item_id_embeddings__values"
      data_type: TYPE_FP64
      dims: -1
      dims: 3072 or 1024, depends on the text embedding dimensions
    }

    ...

    parameters: {
    key: "EXECUTION_ENV_PATH",
    value: {string_value: "/t4rec_23.06.tar.gz"}
    }
    backend: "python"
    ```


# Deploy triton server and ensembled inference workflow

```
docker-compose -f deployment/docker-compose.dev.deploy.yml up
```

# Scripts
We provide two scripts to perform inference on the trained model under two different settings.
- `local_inference.py`: Test locally load trained checkpoints for inference. Supports lora models (peft) and normal models.
- `triton_inference.py`: Test model inference deployed on a remote Triton server. Supports lora models (peft) and normal models.