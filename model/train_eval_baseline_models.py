import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from numba import config
import transformers
from merlin.io import Dataset
sys.path.insert(0, 'transformers4rec') # use our customized transformers4rec
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import MeanReciprocalRankAt, NDCGAt, RecallAt, AvgPrecisionAt
from transformers4rec.torch.masking import CausalLanguageModeling
from model.patches import _pad_across_processes, _compute_masked_targets_mask_last_item
from model.utils import get_nb_trainable_parameters
import wandb

config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

wandb.login()
os.environ["WANDB_PROJECT"] = "predictive_modeling_large"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # save the model weights to W&B
os.environ["WANDB_WATCH"]="false" # turn off watch to log faster


def model(dataset_path="data/processed_nvt_new_data_1226_latest", model_name="bert_baseline"):
    train = Dataset(dataset_path, engine="parquet")
    schema = train.schema

    schema = schema.select_by_name(
       ['item_id-list']
    )

    # Define Next item prediction-task 
    prediction_task = tr.NextItemPredictionTask(weight_tying=True,
                                                sampled_softmax=False,
                                                metrics=[NDCGAt(top_ks=[3, 5, 10], labels_onehot=True),  
                                                RecallAt(top_ks=[3, 5, 10], labels_onehot=True),
                                                AvgPrecisionAt(top_ks=[3, 5, 10], labels_onehot=True),
                                                MeanReciprocalRankAt(top_ks=[3, 5, 10], labels_onehot=True)])


    if model_name == "bert_baseline":

        max_sequence_length, d_model = 110, 1024 
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            embedding_dim_default=1024, # this defines the item embedding dim
            d_output=d_model,
            masking="mlm",
            id_only_baseline=True
        )

        transformer_config = tr.BertConfig.build(d_model=d_model, 
                                                n_head=12, 
                                                n_layer=2, 
                                                intermediate_size=3072,
                                                total_seq_length=max_sequence_length,)
    
    elif model_name == "llama_baseline":

        max_sequence_length, d_model = 110, 2048 
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            embedding_dim_default=1024, # this defines the item embedding dim
            d_output=d_model,
            masking="clm",
            id_only_baseline=True
        )

        transformer_config = tr.LlamaConfig.build(d_model=d_model, 
                                                  n_head=32, 
                                                  n_layer=2, 
                                                  total_seq_length=max_sequence_length,)
    
    elif model_name == "t5_baseline":

        max_sequence_length, d_model = 110, 1024 
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            embedding_dim_default=1024, # this defines the item embedding dim
            d_output=d_model,
            masking="mlm",
            id_only_baseline=True
        )

        transformer_config = tr.T5Config.build(d_model=d_model, 
                                               n_head=8, 
                                               n_layer=2, 
                                               total_seq_length=max_sequence_length,)

    elif model_name == "mixtral_baseline":

        max_sequence_length, d_model = 110, 1024 
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            embedding_dim_default=1024, # this defines the item embedding dim
            d_output=d_model,
            masking="clm",
            id_only_baseline=True
        )

        transformer_config = tr.MixtralConfig.build(hidden_size=d_model, 
                                                    n_head=16, 
                                                    n_layer=2, 
                                                    num_experts_per_tok=2,
                                                    num_local_experts=8,
                                                    total_seq_length=max_sequence_length,
                                                    intermediate_size=3584,) 

    else:
        raise ValueError("model_name not supported")

    model = transformer_config.to_torch_model(input_module, prediction_task)

    print(model)

    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )

    return model, schema, max_sequence_length


def train(model, 
          schema, 
          max_sequence_length, 
          train_batch_size=128, 
          val_batch_size=128,
          train_paths="data/processed_nvt_new_data_1226_latest/train_new",
          val_paths="data/processed_nvt_new_data_1226_latest/val_new"):

    BATCH_SIZE_TRAIN = int(os.environ.get("BATCH_SIZE_TRAIN", train_batch_size))
    BATCH_SIZE_VALID = int(os.environ.get("BATCH_SIZE_VALID", val_batch_size))

    training_args = tr.trainer.T4RecTrainingArguments(
                output_dir="./tmp_test",
                overwrite_output_dir = True,
                max_sequence_length=max_sequence_length,
                data_loader_engine='merlin',
                num_train_epochs=10, # 20 semms to be enough for llama
                dataloader_drop_last=False,
                per_device_train_batch_size = BATCH_SIZE_TRAIN,
                per_device_eval_batch_size = BATCH_SIZE_VALID,
                # learning_rate=2e-5,
                learning_rate=1e-4,
                fp16=True,
                logging_steps=100,
                use_legacy_prediction_loop=False,
                # report_to = ["wandb"],
                report_to=[],
                run_name="mlm_bert_base_1024_all_stuff_no_LLMembed_feat_0122_latest",
                evaluation_strategy = 'steps',
                save_steps=20000, # should be consistent with eval steps
                eval_steps=20000,
                save_total_limit=1,      
                load_best_model_at_end=True,
                metric_for_best_model="eval_/loss", # costomized by t4rec
                greater_is_better=False, # lower loss is better
                save_safetensors=False, # otherwise got shared memory error
            )
    # add missing function to Trainer (BUG)
    tr.Trainer._pad_across_processes = _pad_across_processes
    # this should be correct (BUG)
    CausalLanguageModeling._compute_masked_targets = _compute_masked_targets_mask_last_item

    early_stopping_callback = transformers.EarlyStoppingCallback(
        early_stopping_patience=10, 
    )
    call_backs = [early_stopping_callback]

    recsys_trainer = tr.Trainer(
        model=model,
        args=training_args,
        schema=schema,
        compute_metrics=True,
        callbacks=call_backs,)


    recsys_trainer.train_dataset_or_path = train_paths
    recsys_trainer.eval_dataset_or_path = val_paths
    recsys_trainer.reset_lr_scheduler()
    recsys_trainer.train()


    eval_metrics = recsys_trainer.evaluate(metric_key_prefix="eval")
    print("\n***** Evaluation results ")
    for key in sorted(eval_metrics.keys()):
        if "at_" in key:
            print(" %s = %s" % (key.replace("_at_", "@"), str(eval_metrics[key])))


if __name__ == "__main__":

    tr_model, schema, max_sequence_length = model(dataset_path="data/processed_nvt_new_data_1226_latest",
                                                  model_name="bert_baseline",)
    train(tr_model, 
          schema, 
          max_sequence_length)

