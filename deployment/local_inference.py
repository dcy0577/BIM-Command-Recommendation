import math
import sys
import time
from typing import Dict, List, Optional
import torch
sys.path.insert(0, 'transformers4rec')
from transformers4rec import torch as tr
from transformers4rec.torch.masking import CausalLanguageModeling
from peft import PeftModel
from transformers.trainer_utils import speed_metrics, PredictionOutput
from model.train_eval_full_models import model
from model.patches import _pad_across_processes, _compute_masked_targets_mask_last_item


def evaluate_change_output(
        self,
        eval_dataset = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)


def local_inference():

    tr_model, schema, max_sequence_length, _, val_data_loader = model(dataset_path="data/processed_nvt_new_data_1226_latest", 
                                                    model_name = "mixtral",
                                                    val_batch_size=2,
                                                    val_paths="sub_trainset_for_att_visualization.parquet",)
    
    if isinstance(tr_model, PeftModel):
        peft_model = PeftModel.from_pretrained(
                tr_model,
                model_id = "tmp_clm_llama/checkpoint-800/adapter_model",
                # torch_dtype=torch.float16,
                )
    
        print(peft_model)
    else:
        # load the model from the disk  
        tr_model.load_state_dict(torch.load("mlm_mixtral_2_layers_all_stuff/checkpoint-520000/pytorch_model.bin"))

    tr_model.top_k = 5

    tr_model.eval()
    tr_model.to("cuda:0")

    training_args = tr.trainer.T4RecTrainingArguments(
            output_dir="tmp",
            # per_device_eval_batch_size=2,
            max_sequence_length=max_sequence_length,
            fp16=True,
            report_to=[],
        )
    
    tr.Trainer._pad_across_processes = _pad_across_processes
    tr.Trainer.evaluate = evaluate_change_output
    # this should be correct (BUG)
    CausalLanguageModeling._compute_masked_targets = _compute_masked_targets_mask_last_item

    trainer = tr.Trainer(
        model=tr_model,
        args=training_args,
        schema=schema,
        # test_dataloader=test_loader,
        eval_dataloader=val_data_loader,
        compute_metrics=True,
        )

    trainer.args.predict_top_k = 5
    
    prediction = trainer.evaluate()
    print("=========evaluation===============")
    print(prediction)

if __name__ == "__main__":

    local_inference()