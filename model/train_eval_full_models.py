import sys
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from numba import config
from merlin.io import Dataset
import transformers
sys.path.insert(0, 'transformers4rec')
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt, MeanReciprocalRankAt
from transformers4rec.torch.masking import CausalLanguageModeling
from model.patches import _pad_across_processes, _compute_masked_targets_mask_last_item
from model.utils import SavePeftModelCallback, get_nb_trainable_parameters
import wandb
from peft import PeftModel
from merlin.dataloader.ops.embeddings import EmbeddingOperator
from transformers4rec.torch.utils.data_utils import MerlinDataLoader

config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

wandb.login()
os.environ["WANDB_PROJECT"] = "predictive_modeling_large"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # save the model weights to W&B
os.environ["WANDB_WATCH"]="false" # turn off watch to log faster


def model(dataset_path="data/processed_nvt_new_data/part_0.parquet", 
          model_name = "mixtral", 
          train_bacth_size=128,
          val_batch_size=128,
          train_paths="data/preproc_sessions_whole_new_data_1226_latest/train_new",
          val_paths="data/preproc_sessions_whole_new_data_1226_latest/val_new",
          pretrained_emb_path="data/pre-trained-item-id-new-data_0122.npy"):
    
    # this is just for getting the schema
    train = Dataset(dataset_path, engine="parquet")
    schema = train.schema

    schema = schema.select_by_name(
   ['item_id-list', 'classification-list', 'target-list', 'merge_count_norm-list', 'timestamp_interval_norm_global-list'] # 'timestamp_interval_norm_global-list' 'cat-list'
)

    item_cardinality = schema["item_id-list"].int_domain.max + 1
    np_emb_item_id = np.load(pretrained_emb_path) # (item_cardinality, pretrained_dim)  # 3072 dims
    embeddings_op = EmbeddingOperator(
        np_emb_item_id, 
        lookup_key="item_id-list", 
        embedding_name="pretrained_item_id_embeddings",
        # mmap=True,
    )


    # set dataloader with pre-trained embeddings
    # with this approach we cant use the embedding projection function as we dont have the embedding tags!!!
    # so the pretrained_output_dims and normalizer doesnt work!!!
    data_loader = MerlinDataLoader.from_schema(
        schema,
        Dataset(train_paths, schema=schema, engine="parquet"),
        max_sequence_length=110,
        batch_size=train_bacth_size,
        transforms=[embeddings_op],
        shuffle=True,
    )

    
    val_data_loader = MerlinDataLoader.from_schema(
        schema,
        Dataset(val_paths, schema=schema, engine="parquet"),
        max_sequence_length=110,
        batch_size=val_batch_size,
        transforms=[embeddings_op],
        shuffle=False,
    )
    
    model_schema = data_loader.output_schema
    # Define Next item prediction-task 
    prediction_task = tr.NextItemPredictionTask(weight_tying=False, # we use linear layer for prediction
                                                sampled_softmax=False, # we use full softmax
                                                metrics=[NDCGAt(top_ks=[3, 5, 10], labels_onehot=True),  
                                                RecallAt(top_ks=[3, 5, 10], labels_onehot=True),
                                                MeanReciprocalRankAt(top_ks=[3, 5, 10], labels_onehot=True)])

    if model_name == "llama":
        max_sequence_length, d_model = 110, 2048 
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            model_schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=1024,
            embedding_dim_default=1024, # this defines the default all categorical feature embedding dim
            embedding_dims= {"item_id-list": 1024}, # this defines the item embedding dim
            d_output=d_model,
            masking="clm",
            multi_task_labels=True,
            custom_aggregation=True,
            self_attention_agg=True,
            att_pooling=True,
        )

        transformer_config = tr.LlamaConfig.build(hidden_size=d_model,
                                                   n_head=32, 
                                                   n_layer=2, 
                                                   total_seq_length=max_sequence_length, 
                                                   )
    
    elif model_name == "llama_lora":
        max_sequence_length, d_model = 110, 4096 # llama2 7B
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            model_schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=1024,
            embedding_dim_default=1024, # this defines the default all categorical feature embedding dim
            embedding_dims= {"item_id-list": 1024}, # this defines the item embedding dim
            d_output=d_model,
            masking="clm",
            multi_task_labels=True,
            custom_aggregation=True,
            self_attention_agg=True,
            att_pooling=True,
        )

        transformer_config = tr.LlamaConfig.build(hidden_size=d_model, 
                                                    n_head=32, 
                                                    n_layer=32, 
                                                    total_seq_length=max_sequence_length, 
                                                    if_pretrained=True,
                                                    use_lora=True,)

    
    elif model_name == "t5": 
        max_sequence_length, d_model = 110, 1024 
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            model_schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=1024,
            embedding_dim_default=1024, # this defines the default all categorical feature embedding dim
            embedding_dims= {"item_id-list": 1024}, #  has to be consist with d_model for decoder input
            d_output=d_model,
            masking="mlm",
            multi_task_labels=True,
            custom_aggregation=True,
            self_attention_agg=True,
            att_pooling=True,
        )

        transformer_config = tr.T5Config.build(d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length)

    
    elif model_name == "mixtral":
        max_sequence_length, d_model = 110, 1024 #1536 original egice setting
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            model_schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=1024,
            embedding_dim_default=1024, # this defines the default all categorical feature embedding dim
            embedding_dims= {"item_id-list": 1024}, # this defines the item embedding dim
            d_output=d_model,
            masking="clm",
            multi_task_labels=True,
            custom_aggregation=True,
            self_attention_agg=True,
            att_pooling=True,
        )
        
        transformer_config = tr.MixtralConfig.build(hidden_size=d_model, 
                                                n_head=16, 
                                                n_layer=2, 
                                                num_experts_per_tok=2,
                                                num_local_experts=8,
                                                total_seq_length=max_sequence_length,
                                                intermediate_size=3584, # 14336 # 7168
                                                )
        
    
    elif model_name == "bert":
        max_sequence_length, d_model = 110, 1024 # 1024
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            model_schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=1024,
            embedding_dim_default=1024, # this defines the default all categorical feature embedding dim
            embedding_dims= {"item_id-list": 1024}, # this defines the item embedding dim
            d_output=d_model,
            masking="mlm",
            multi_task_labels=True,
            custom_aggregation=True,
            self_attention_agg=True,
            att_pooling=True,
        )

        transformer_config = tr.BertConfig.build(d_model=d_model, 
                                                 n_head=16, 
                                                 n_layer=2, 
                                                 intermediate_size=4096, 
                                                 total_seq_length=max_sequence_length)
        

    elif model_name == "bert_base":
        max_sequence_length, d_model = 110, 768 # bert base : 12 layers, 12 heads, 768 hidden size, intermediate size 3072
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            model_schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=1024,
            embedding_dim_default=1024, # this defines the default all categorical feature embedding dim
            embedding_dims= {"item_id-list": 1024}, # this defines the item embedding dim
            d_output=d_model,
            masking="mlm",
            multi_task_labels=True,
            custom_aggregation=True,
            self_attention_agg=True,
            att_pooling=True,
        )

        transformer_config = tr.BertConfig.build( d_model=d_model, 
                                                 n_head=12, 
                                                 n_layer=12, 
                                                 intermediate_size=3072, 
                                                 total_seq_length=max_sequence_length,
                                                 if_pretrained=True,)
    
    elif model_name == "bert_large":   
        max_sequence_length, d_model = 110, 1024 # bert large : 24 layers, 16 heads, 1024 hidden size, intermediate size 4096
        # Define input module to process tabular input-features and to prepare masked inputs
        input_module = tr.TabularSequenceFeatures.from_schema(
            model_schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=1024,
            embedding_dim_default=1024, # this defines the default all categorical feature embedding dim
            embedding_dims= {"item_id-list": 1024}, # this defines the item embedding dim
            d_output=d_model,
            masking="mlm",
            multi_task_labels=True,
            custom_aggregation=True,
            self_attention_agg=True,
            att_pooling=True,
        )

        transformer_config = tr.BertConfig.build( d_model=d_model, 
                                                 n_head=16, 
                                                 n_layer=24, 
                                                 intermediate_size=4096, 
                                                 total_seq_length=max_sequence_length,
                                                 if_pretrained=True,)


    else:
        raise ValueError("model_name not found")
    
    model = transformer_config.to_torch_model(input_module, prediction_task)
        
    print(model)

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()
    else:
        trainable_params, all_param = get_nb_trainable_parameters(model)
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    return model, schema, max_sequence_length, data_loader, val_data_loader
    

def train(model, schema, max_sequence_length, data_loader, val_data_loader):

    training_args = tr.trainer.T4RecTrainingArguments(
                output_dir="./tmp_test",
                overwrite_output_dir = True,
                max_sequence_length=max_sequence_length,
                data_loader_engine='merlin',
                num_train_epochs=10, 
                dataloader_drop_last=False,
                learning_rate=3e-5, 
                fp16=True,
                fp16_full_eval=False, # ture will reduce cuda memory
                logging_steps=100,
                use_legacy_prediction_loop=False,
                # report_to = ["wandb"],
                report_to = [],
                run_name="llama_all_stuff_att_ce_loss_0122_newdata",
                evaluation_strategy = 'steps',
                save_strategy='steps', # for lora
                save_steps=20000, # should be consistent with eval steps
                eval_steps=20000,
                save_total_limit=1,
                load_best_model_at_end=True,  # not working for peft model
                metric_for_best_model="eval_/loss", # costomized by t4rec
                greater_is_better=False,
                use_cpu=False,
                save_safetensors=False, # otherwise got shared memory error
            )
    
    # add missing function to Trainer (BUG)
    tr.Trainer._pad_across_processes = _pad_across_processes
    # this should be correct (BUG)
    CausalLanguageModeling._compute_masked_targets = _compute_masked_targets_mask_last_item


    early_stopping_callback = transformers.EarlyStoppingCallback(
        early_stopping_patience=10 
    )

    if isinstance(model, PeftModel):
        call_backs = [early_stopping_callback, SavePeftModelCallback]
    else:
        call_backs = [early_stopping_callback]

    recsys_trainer = tr.Trainer(
        model=model,
        args=training_args,
        schema=schema,
        train_dataloader=data_loader,
        eval_dataloader=val_data_loader,
        compute_metrics=True,
        callbacks=call_backs,)
    
    recsys_trainer.reset_lr_scheduler()

    recsys_trainer.train()

    # final evaluation
    eval_metrics = recsys_trainer.evaluate(metric_key_prefix="eval")
    print("\n***** Evaluation results ")
    for key in sorted(eval_metrics.keys()):
        if "at_" in key:
            print(" %s = %s" % (key.replace("_at_", "@"), str(eval_metrics[key])))


if __name__ == "__main__":

    tr_model, schema, max_sequence_length, dataloader, val_dataloader = model(dataset_path="data/processed_nvt_new_data_1226_latest", 
                                                                              model_name = "llama",
                                                                              train_paths="data/preproc_sessions_whole_new_data_1226_latest/train_new",
                                                                              val_paths="data/preproc_sessions_whole_new_data_1226_latest/val_new")

    train(tr_model, schema, max_sequence_length, dataloader, val_dataloader)


