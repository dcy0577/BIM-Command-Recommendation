#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import logging
from math import sqrt
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torchmetrics as tm
import torch.nn.functional as F

from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func, MoeModelOutputWithPast

from ..block.base import Block, BuildableBlock, SequentialBlock
from ..block.mlp import MLPBlock
from ..masking import MaskedLanguageModeling
from ..ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt
from ..utils.torch_utils import LambdaModule
from .base import BlockType, PredictionTask


LOG = logging.getLogger("transformers4rec")


class BinaryClassificationPrepareBlock(BuildableBlock):
    """Prepares the output layer of the binary classification prediction task.
    The output layer is a SequentialBlock of a torch linear
    layer followed by a sigmoid activation and a squeeze operation.
    """

    def build(self, input_size) -> SequentialBlock:
        """Builds the output layer of binary classification based on the input_size.

        Parameters
        ----------
        input_size: Tuple[int]
            The size of the input tensor, specifically the last dimension is
            used for setting the input dimension of the linear layer.

        Returns
        -------
        SequentialBlock
            A SequentialBlock consisting of a linear layer (with input dimension equal to the last
            dimension of input_size), a sigmoid activation, and a squeeze operation.
        """
        return SequentialBlock(
            torch.nn.Linear(input_size[-1], 1, bias=False),
            torch.nn.Sigmoid(),
            LambdaModule(lambda x: torch.squeeze(x, -1)),
            output_size=[
                None,
            ],
        )


class BinaryClassificationTask(PredictionTask):
    """Returns a ``PredictionTask`` for binary classification.

    Example usage::

        # Define the input module to process the tabular input features.
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=d_model,
            aggregation="concat",
            masking=None,
        )

        # Define XLNetConfig class and set default parameters for HF XLNet config.
        transformer_config = tr.XLNetConfig.build(
            d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
        )

        # Define the model block including: inputs, masking, projection and transformer block.
        body = tr.SequentialBlock(
            input_module,
            tr.MLPBlock([64]),
            tr.TransformerBlock(
                transformer_config,
                masking=input_module.masking
            )
        )

        # Define a head with BinaryClassificationTask.
        head = tr.Head(
            body,
            tr.BinaryClassificationTask(
                "click",
                summary_type="mean",
                metrics=[
                    tm.Precision(task='binary'),
                    tm.Recall(task='binary'),
                    tm.Accuracy(task='binary'),
                    tm.F1Score(task='binary')
                ]
            ),
            inputs=input_module,
        )

        # Get the end-to-end Model class.
        model = tr.Model(head)

    Parameters
    ----------

    target_name: Optional[str] = None
        Specifies the variable name that represents the positive and negative values.

    task_name: Optional[str] = None
        Specifies the name of the prediction task. If this parameter is not specified,
        a name is automatically constructed based on ``target_name`` and the Python
        class name of the model.

    task_block: Optional[BlockType] = None
        Specifies a module to transform the input tensor before computing predictions.

    loss: torch.nn.Module
        Specifies the loss function for the task.
        The default class is ``torch.nn.BCELoss``.

    metrics: Tuple[torch.nn.Module, ...]
        Specifies the metrics to calculate during training and evaluation.
        The default metrics are ``Precision``, ``Recall``, and ``Accuracy``.

    summary_type: str
        Summarizes a sequence into a single tensor. Accepted values are:

            - ``last`` -- Take the last token hidden state (like XLNet)
            - ``first`` -- Take the first token hidden state (like Bert)
            - ``mean`` -- Take the mean of all tokens hidden states
            - ``cls_index`` -- Supply a Tensor of classification token position (GPT/GPT-2)
            - ``attn`` -- Not implemented now, use multi-head attention
    """

    DEFAULT_LOSS = torch.nn.BCELoss()
    DEFAULT_METRICS = (
        tm.Precision(num_classes=2, task="binary"),
        tm.Recall(num_classes=2, task="binary"),
        tm.Accuracy(task="binary"),
        # TODO: Fix this: tm.AUC()
    )

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[BlockType] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        self.target_dim = 1
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            pre=BinaryClassificationPrepareBlock(),
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
        )


class RegressionPrepareBlock(BuildableBlock):
    """Prepares the output layer of the regression prediction task.
    The output layer is a SequentialBlock of a torch linear
    layer followed by a squeeze operation.
    """

    def build(self, input_size) -> SequentialBlock:
        """Builds the output layer of regression based on the input_size.

        Parameters
        ----------
        input_size: Tuple[int]
            The size of the input tensor, specifically the last dimension is
            used for setting the input dimension of the linear layer.

        Returns
        -------
        SequentialBlock
            A SequentialBlock consisting of a linear layer (with input dimension equal to
            the last dimension of input_size), and a squeeze operation.
        """
        return SequentialBlock(
            torch.nn.Linear(input_size[-1], 1),
            LambdaModule(lambda x: torch.squeeze(x, -1)),
            output_size=[
                None,
            ],
        )


class RegressionTask(PredictionTask):
    """Returns a ``PredictionTask`` for regression.

    Example usage::

        # Define the input module to process the tabular input features.
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=d_model,
            aggregation="concat",
            masking=None,
        )

        # Define XLNetConfig class and set default parameters for HF XLNet config.
        transformer_config = tr.XLNetConfig.build(
            d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
        )

        # Define the model block including: inputs, projection and transformer block.
        body = tr.SequentialBlock(
            input_module,
            tr.MLPBlock([64]),
            tr.TransformerBlock(
                transformer_config,
            )
        )

        # Define a head with BinaryClassificationTask.
        head = tr.Head(
            body,
            tr.RegressionTask(
                "watch_time",
                summary_type="mean",
                metrics=[tm.regression.MeanSquaredError()]
            ),
            inputs=input_module,
        )

        # Get the end-to-end Model class.
        model = tr.Model(head)

    Parameters
    ----------

    target_name: Optional[str]
        Specifies the variable name that represents the continuous value to predict.
        By default None

    task_name: Optional[str]
        Specifies the name of the prediction task. If this parameter is not specified,
        a name is automatically constructed based on ``target_name`` and the Python
        class name of the model.
        By default None

    task_block: Optional[BlockType] = None
        Specifies a module to transform the input tensor before computing predictions.

    loss: torch.nn.Module
        Specifies the loss function for the task.
        The default class is ``torch.nn.MSELoss``.

    metrics: Tuple[torch.nn.Module, ...]
        Specifies the metrics to calculate during training and evaluation.
        The default metric is MeanSquaredError.

    summary_type: str
        Summarizes a sequence into a single tensor. Accepted values are:

            - ``last`` -- Take the last token hidden state (like XLNet)
            - ``first`` -- Take the first token hidden state (like Bert)
            - ``mean`` -- Take the mean of all tokens hidden states
            - ``cls_index`` -- Supply a Tensor of classification token position (GPT/GPT-2)
            - ``attn`` -- Not implemented now, use multi-head attention
    """

    DEFAULT_LOSS = torch.nn.MSELoss()
    DEFAULT_METRICS = (tm.regression.MeanSquaredError(),)

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[BlockType] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        self.target_dim = 1
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            pre=RegressionPrepareBlock(),
        )


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=0.9, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  #  False Positives weight
        self.beta = beta    #  False Negatives weight
        self.smooth = smooth  # smoothing factor to avoid division by zero
    # this is very sensitive to class amount
    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        # make input to be in range [0, 1]
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).float()
        tempreture = 0.1
        probs = F.softmax(inputs/tempreture, dim=1)
        prob_sums = probs.sum(dim=1)

        # probs = probs.transpose(0, 1) # [num_classes, batch_size]
        # targets_one_hot = targets_one_hot.transpose(0, 1) # [num_classes, batch_size]

        TP = (probs * targets_one_hot).sum(dim=0)
        FP = (probs * (1 - targets_one_hot)).sum(dim=0)
        FN = ((1 - probs) * targets_one_hot).sum(dim=0)

        # Tversky index
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        loss = 1 - Tversky
        return loss.mean()
    
class NextItemPredictionTask(PredictionTask):
    """This block performs item prediction task for session and sequential-based models.
    It requires a body containing a masking schema to use for training and target generation.
    For the supported masking schemes, please refers to:
    https://nvidia-merlin.github.io/Transformers4Rec/stable/model_definition.html#sequence-masking

    Parameters
    ----------
    loss: torch.nn.Module
        Loss function to use. Defaults to NLLLos.
    metrics: Iterable[torchmetrics.Metric]
        List of ranking metrics to use for evaluation.
    task_block:
        Module to transform input tensor before computing predictions.
    task_name: str, optional
        Name of the prediction task, if not provided a name will be automatically constructed based
        on the target-name & class-name.
    weight_tying: bool
        The item id embedding table weights are shared with the prediction network layer.
    softmax_temperature: float
        Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
        Value 1.0 reduces to regular softmax.
    padding_idx: int
        pad token id.
    target_dim: int
        vocabulary size of item ids
    sampled_softmax: Optional[bool]
        Enables sampled softmax. By default False
    max_n_samples: Optional[int]
        Number of samples for sampled softmax. By default 100
    """

    DEFAULT_METRICS = (
        # default metrics suppose labels are int encoded
        NDCGAt(top_ks=[10, 20], labels_onehot=True),
        AvgPrecisionAt(top_ks=[10, 20], labels_onehot=True),
        RecallAt(top_ks=[10, 20], labels_onehot=True),
    )

    def __init__(
        self,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        metrics: Iterable[tm.Metric] = DEFAULT_METRICS,
        task_block: Optional[BlockType] = None,
        task_name: str = "next-item",
        weight_tying: bool = False,
        softmax_temperature: float = 1,
        padding_idx: int = 0,
        target_dim: int = None,
        sampled_softmax: Optional[bool] = False,
        max_n_samples: Optional[int] = 100,
    ):
        super().__init__(loss=loss, metrics=metrics, task_block=task_block, task_name=task_name)
        self.softmax_temperature = softmax_temperature
        self.weight_tying = weight_tying
        self.padding_idx = padding_idx
        self.target_dim = target_dim
        self.sampled_softmax = sampled_softmax
        self.max_n_samples = max_n_samples

        self.item_embedding_table = None
        self.masking = None

    def build(self, body, input_size, device=None, inputs=None, task_block=None, pre=None):
        """Build method, this is called by the `Head`."""
        # print(body)
        if not len(input_size) == 3 or isinstance(input_size, dict):
            raise ValueError(
                "NextItemPredictionTask needs a 3-dim vector as input, found:" f"{input_size}"
            )

        # Retrieve the embedding module to get the name of itemid col and its related table
        if not inputs:
            inputs = body.inputs
        if not getattr(inputs, "item_id", None):
            raise ValueError(
                "For Item Prediction task a categorical_module "
                "including an item_id column is required."
            )
        self.embeddings = inputs.categorical_module
        if not self.target_dim:
            self.target_dim = self.embeddings.item_embedding_table.num_embeddings
        if self.weight_tying:
            self.item_embedding_table = self.embeddings.item_embedding_table
            item_dim = self.item_embedding_table.weight.shape[1]
            if input_size[-1] != item_dim and not task_block:
                LOG.warning(
                    f"Projecting inputs of NextItemPredictionTask to'{item_dim}' "
                    f"As weight tying requires the input dimension '{input_size[-1]}' "
                    f"to be equal to the item-id embedding dimension '{item_dim}'"
                )
                # project input tensors to same dimension as item-id embeddings
                task_block = MLPBlock([item_dim], activation=None)

        # Retrieve the masking from the input block
        self.masking = inputs.masking
        if not self.masking:
            raise ValueError(
                "The input block should contain a masking schema for training and evaluation"
            )
        self.padding_idx = self.masking.padding_idx
        pre = NextItemPredictionPrepareBlock(
            target_dim=self.target_dim,
            weight_tying=self.weight_tying,
            item_embedding_table=self.item_embedding_table,
            softmax_temperature=self.softmax_temperature,
            sampled_softmax=self.sampled_softmax,
            max_n_samples=self.max_n_samples,
            min_id=self.padding_idx + 1,
        )

        use_class_weight = False
        # if use_class_weight:
        #     classificaiton_weight = compute_class_weights(parquet_path="categories/unique.classification.parquet").to("cuda:0")
        #     target_weight = compute_class_weights(parquet_path="categories/unique.target.parquet").to("cuda:0")
        #     item_id_weight = compute_class_weights(parquet_path="categories/unique.item_id.parquet").to("cuda:0")
        # New: lets first hardcode the traget dim for 2 additional category labels
        # have to first build the model before check whether the additional category labels are provided
        # this is for classification
        # when init the model, this field is by default None, so we need to hardcode it here
        multi_task = body.inputs.multi_task_labels
        if multi_task:
            # self.cat_loss1 = torch.nn.CrossEntropyLoss(weight=classificaiton_weight if use_class_weight else None)
            self.cat_loss1 = FocalLoss(gamma=2, reduction='mean')
            # this is for target which is unbalanced
            self.cat_loss2 = FocalLoss(gamma=2, reduction='mean')
            # self.cat_loss2 = torch.nn.CrossEntropyLoss()
            # this is for classification, for output layer we dont need relu, the softmax is buit in the loss function
            self.mlp_branch1 = MLPBlock([177], use_bias=False, activation=None).build([input_size[-1] if not task_block else item_dim])
            # this is for target
            self.mlp_branch2 = MLPBlock([366], use_bias=False, activation=None).build([input_size[-1]if not task_block else item_dim])

            # this is for cat
            # self.cat_loss3= FocalLoss(alpha=classificaiton_weight if use_class_weight else None, gamma=2, reduction='mean')
            # self.mlp_branch3 = MLPBlock([7], use_bias=False, activation=None).build([input_size[-1] if not task_block else item_dim])

        # lets try focal loss for item id as well
        self.loss = FocalLoss(gamma=2, reduction='mean')

        # lets try tversky loss for item id as well
        # self.loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-6)

        # this is for merge_count_norm
        # self.cat_loss4 = torch.nn.MSELoss()
        # self.mlp_branch4 = MLPBlock([1], use_bias=False, activation=None).build([input_size[-1] if not task_block else item_dim])

        # better to access the transformer config here
        self.transformer_name = body[1].transformer.config.model_type # "mixtral"
        if self.transformer_name == "mixtral":
            self.router_aux_loss_coef=body[1].transformer.config.router_aux_loss_coef
            self.num_experts = body[1].transformer.config.num_local_experts
            self.num_experts_per_tok = body[1].transformer.config.num_experts_per_tok

        super().build(
            body, input_size, device=device, inputs=inputs, task_block=task_block, pre=pre
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets=None,
        training=False,
        testing=False,
        top_k=None,
        **kwargs,
    ):
        
        aux_loss = None
        if training and self.transformer_name == "mixtral":
            # need to compute the auxiliary loss for balancing experts
            aux_loss = load_balancing_loss_func(
                inputs[-1], # router_logits [batch_size X sequence_length, num_experts]
                self.num_experts,
                self.num_experts_per_tok,
                None,
            )
        
        if isinstance(inputs, (tuple, list, MoeModelOutputWithPast)):
            inputs = inputs[0]

        # this x is output from the transformer blocks
        x = inputs.float()

        if self.task_block:
            x = self.task_block(x)  # type: ignore

        # Retrieve labels from masking
        if training or testing:
            labels = self.masking.masked_targets  # type: ignore
            trg_flat = labels.flatten()
            non_pad_mask = trg_flat != self.padding_idx
            labels_all = torch.masked_select(trg_flat, non_pad_mask).long()
            # remove padded items, keep only masked positions
            x = self.remove_pad_3d(x, non_pad_mask)
            y = labels_all
            x_all, y = self.pre(x, targets=y, training=training, testing=testing)  # type: ignore
            # Compute loss here
            loss = self.loss(x_all, y)

            if aux_loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

            # new: lets do the same for the 2 additional category labels
            # we want do multi task, we need corresponding y and add the multiple self.loss up
            if self.masking.additional_targets_labels:

                # -----------------llm augmented multi task-----------------
                labels1 = self.masking.additional_targets_labels["classification-list"]
                labels2 = self.masking.additional_targets_labels["target-list"]
                # labels3 = self.masking.additional_targets_labels["cat-list"]
                trg_flat1 = labels1.flatten()
                trg_flat2 = labels2.flatten()
                # trg_flat3 = labels3.flatten()
                non_pad_mask1 = trg_flat1 != self.padding_idx
                non_pad_mask2 = trg_flat2 != self.padding_idx
                # non_pad_mask3 = trg_flat3 != self.padding_idx
                labels_all1 = torch.masked_select(trg_flat1, non_pad_mask1).long()
                labels_all2 = torch.masked_select(trg_flat2, non_pad_mask2).long()
                # labels_all3 = torch.masked_select(trg_flat3, non_pad_mask3).long()
                y1 = labels_all1
                y2 = labels_all2
                # y3 = labels_all3
                x1 = self.mlp_branch1(x)
                x2 = self.mlp_branch2(x)
                # x3 = self.mlp_branch3(x)
                loss1 = self.cat_loss1(x1, y1)
                loss2 = self.cat_loss2(x2, y2)
                # loss3 = self.cat_loss3(x3, y3)

                gama1 = 0.5
                gama2 = 0.5
                gamma3 = 0.9
                total_loss = loss + gama1*loss1 + gama2*loss2 # + gamma3*loss3
                return {
                    "loss": total_loss,
                    "labels": y,
                    "predictions": x_all,
                }
                # -----------------llm augmented multi task-----------------

                # workflow for multi task
                # labels3 = self.masking.additional_targets_labels["cat-list"]
                # labels4 = self.masking.additional_targets_labels["merge_count_norm-list"]
                # trg_flat3 = labels3.flatten()
                # trg_flat4 = labels4.flatten()
                # non_pad_mask3 = trg_flat3 != self.padding_idx
                # non_pad_mask4 = trg_flat4 != self.padding_idx
                # labels_all3 = torch.masked_select(trg_flat3, non_pad_mask3).long()
                # labels_all4 = torch.masked_select(trg_flat4, non_pad_mask4).float()
                # y3 = labels_all3
                # y4 = labels_all4
                # x3 = self.mlp_branch3(x)
                # x4 = self.mlp_branch4(x)
                # loss3 = self.cat_loss3(x3, y3)
                # x4 = x4.view(-1) # flatten the output
                # loss4 = self.cat_loss4(x4, y4)

                # gama3 = 0.9
                # gama4 = 0.9
                # total_loss = loss + gama3*loss3 + gama4*loss4
                # return {
                #     "loss": total_loss,
                #     "labels": y,
                #     "predictions": x_all,
                # }

            return {
                "loss": loss,
                "labels": y,
                "predictions": x_all,
            }
        else:
            # Get the hidden position to use for predicting the next item
            labels = self.embeddings.item_seq
            non_pad_mask = labels != self.padding_idx
            rows_ids = torch.arange(labels.size(0), dtype=torch.long, device=labels.device)
            if isinstance(self.masking, MaskedLanguageModeling):
                last_item_sessions = non_pad_mask.sum(dim=1)
            else:
                last_item_sessions = non_pad_mask.sum(dim=1) - 1
            x = x[rows_ids, last_item_sessions]

            # Compute predictions probs
            x, _ = self.pre(x)  # type: ignore

            if top_k is None:
                return x
            else:
                preds_sorted_item_scores, preds_sorted_item_ids = torch.topk(x, k=top_k, dim=-1)
                return preds_sorted_item_scores, preds_sorted_item_ids

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor

    def calculate_metrics(self, predictions, targets) -> Dict[str, torch.Tensor]:  # type: ignore
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        outputs = {}
        predictions = self.forward_to_prediction_fn(predictions)

        for metric in self.metrics:
            result = metric(predictions, targets)
            outputs[self.metric_name(metric)] = result

        return outputs

    def compute_metrics(self):
        metrics = {
            self.metric_name(metric): metric.compute()
            for metric in self.metrics
            if getattr(metric, "top_ks", None)
        }
        # Explode metrics for each cut-off
        # TODO make result generic:
        # To accept a mix of ranking metrics and others not requiring top_ks ?
        topks = {self.metric_name(metric): metric.top_ks for metric in self.metrics}
        results = {}
        for name, metric in metrics.items():
            # Fix for when using a single cut-off, as torch metrics convert results to scalar
            # when a single element vector is returned
            if len(metric.size()) == 0:
                metric = metric.unsqueeze(0)
            for measure, k in zip(metric, topks[name]):
                results[f"{name}_{k}"] = measure
        return results


class NextItemPredictionPrepareBlock(BuildableBlock):
    """Prepares the output layer of the next item prediction task.
    The output layer is a an instance of `_NextItemPredictionTask` class.

    Parameters
    ----------
    target_dim: int
        The output dimension for next-item predictions.
    weight_tying: bool, optional
        If true, ties the weights of the prediction layer and the item embedding layer.
        By default False.
    item_embedding_table: torch.nn.Module, optional
        The module containing the item embedding table.
        By default None.
    softmax_temperature: float, optional
        The temperature to be applied to the softmax function. Defaults to 0.
    sampled_softmax: bool, optional
        If true, sampled softmax is used for approximating the full softmax function.
        By default False.
    max_n_samples: int, optional
        The maximum number of samples when using sampled softmax.
        By default 100.
    min_id: int, optional
        The minimum value of the range for the log-uniform sampling.
        By default 0.
    """

    def __init__(
        self,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
        sampled_softmax: Optional[bool] = False,
        max_n_samples: Optional[int] = 100,
        min_id: Optional[int] = 0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature
        self.sampled_softmax = sampled_softmax
        self.max_n_samples = max_n_samples
        self.min_id = min_id

    def build(self, input_size) -> Block:
        """Builds the output layer of next-item prediction based on the input_size.

        Parameters
        ----------
        input_size : Tuple[int]
            The size of the input tensor, specifically the last dimension is
            used for setting the input dimension of the output layer.
        Returns
        -------
        Block[_NextItemPredictionTask]
            an instance of _NextItemPredictionTask
        """
        return Block(
            _NextItemPredictionTask(
                input_size,
                self.target_dim,
                self.weight_tying,
                self.item_embedding_table,
                self.softmax_temperature,
                self.sampled_softmax,
                self.max_n_samples,
                self.min_id,
            ),
            [-1, self.target_dim],
        )


class _NextItemPredictionTask(torch.nn.Module):
    """Predict the interacted item-id probabilities.

    - During inference, the task consists of predicting the next item.
    - During training, the class supports the following Language modeling tasks:
        Causal LM, Masked LM, Permutation LM and Replacement Token Detection

    Parameters:
    -----------
        input_size: int
            Input size of this module.
        target_dim: int
            Dimension of the target.
        weight_tying: bool
            The item id embedding table weights are shared with the prediction network layer.
        item_embedding_table: torch.nn.Module
            Module that's used to store the embedding table for the item.
        softmax_temperature: float
            Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
            Value 1.0 reduces to regular softmax.
        sampled_softmax: Optional[bool]
            Enables sampled softmax. By default False
        max_n_samples: Optional[int]
            Number of samples for sampled softmax. By default 100
        min_id : Optional[int]
            The minimum value of the range for the log-uniform sampling. By default 0.
    """

    def __init__(
        self,
        input_size: Sequence,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
        sampled_softmax: Optional[bool] = False,
        max_n_samples: Optional[int] = 100,
        min_id: Optional[int] = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature
        self.sampled_softmax = sampled_softmax
        # initialize a pytorch mlp layer
        # self.mlp_head = MLPBlock([698], use_bias=False).build([128])
        self.mlp_head = torch.nn.Linear(input_size[-1],  self.target_dim, bias=False)
        # self.softmax = torch.nn.Softmax(dim=-1)

        if not self.weight_tying:
            if self.mlp_head:
                self.output_layer = self.mlp_head
            else:
                self.output_layer = torch.nn.Parameter(torch.empty(self.target_dim, input_size[-1])) # should be MLP??
                torch.nn.init.kaiming_uniform_(self.output_layer, a=sqrt(5))

        if self.sampled_softmax:
            self.sampler = LogUniformSampler(
                max_n_samples=max_n_samples,
                max_id=target_dim,
                min_id=min_id,
                unique_sampling=True,
            )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        training=False,
        testing=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.weight_tying:
            output_weights = self.item_embedding_table.weight
        else:
            if self.mlp_head:
                output_weights = self.mlp_head(inputs)
                return output_weights, targets
            else:
                output_weights = self.output_layer

        if self.sampled_softmax and training:
            logits, targets = self.sampled(inputs, targets, output_weights)
        else:
            logits = inputs @ output_weights.t()  # type: ignore

        if self.softmax_temperature:
            # Softmax temperature to reduce model overconfidence
            # and better calibrate probs and accuracy
            logits = torch.div(logits, self.softmax_temperature)

        return logits, targets

    def sampled(self, inputs, targets, output_weights):
        """Returns logits using sampled softmax"""
        neg_samples, targets_probs, samples_probs = self.sampler.sample(targets)

        positive_weights = output_weights[targets]
        negative_weights = output_weights[neg_samples]

        positive_scores = (inputs * positive_weights).sum(dim=-1, keepdim=True)
        negative_scores = inputs @ negative_weights.t()

        # logQ correction, to not overpenalize popular items for being sampled
        # more often as negatives
        epsilon = 1e-16
        positive_scores -= torch.unsqueeze(torch.log(targets_probs + epsilon), dim=-1)
        negative_scores -= torch.unsqueeze(torch.log(samples_probs + epsilon), dim=0)

        # Remove accidental matches
        accidental_hits = torch.unsqueeze(targets, -1) == torch.unsqueeze(neg_samples, 0)
        negative_scores[accidental_hits] = torch.finfo(torch.float16).min / 100.0

        logits = torch.cat([positive_scores, negative_scores], axis=1)
        new_targets = torch.zeros(logits.shape[0], dtype=torch.int64, device=targets.device)

        return logits, new_targets

    def _get_name(self) -> str:
        return "NextItemPredictionTask"


class LogUniformSampler(torch.nn.Module):
    def __init__(
        self,
        max_n_samples: int,
        max_id: int,
        min_id: Optional[int] = 0,
        unique_sampling: bool = True,
        n_samples_multiplier_before_unique: int = 2,
    ):
        """LogUniformSampler samples negative samples based on a log-uniform distribution.
        `P(class) = (log(class + 2) - log(class + 1)) / log(max_id + 1)`

        This implementation is based on to:
        https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/log_uniform_sampler.py
        TensorFlow Reference:
        https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py

        LogUniformSampler assumes item ids are sorted decreasingly by their frequency.

        if `unique_sampling==True`, then only unique sampled items will be returned.
        The actual # samples will vary from run to run if `unique_sampling==True`,
        as sampling without replacement (`torch.multinomial(..., replacement=False)`) is slow,
        so we use `torch.multinomial(..., replacement=True).unique()` which doesn't guarantee
        the same number of unique sampled items. You can try to increase
        n_samples_multiplier_before_unique to increase the chances to have more
        unique samples in that case.

        Parameters
        ----------
        max_n_samples : int
            The maximum desired number of negative samples. The number of samples might be
            smaller than that if `unique_sampling==True`, as explained above.
        max_id : int
            The maximum value of the range for the log-uniform distribution.
        min_id : Optional[int]
            The minimum value of the range for the log-uniform sampling. By default 0.
        unique_sampling : bool
            Whether to return unique samples. By default True
        n_samples_multiplier_before_unique : int
            If unique_sampling=True, it is not guaranteed that the number of returned
            samples will be equal to max_n_samples, as explained above.
            You can increase n_samples_multiplier_before_unique to maximize
            chances that a larger number of unique samples is returned.
        """
        super().__init__()

        if max_id <= 0:
            raise ValueError("max_id must be a positive integer.")
        if max_n_samples <= 0:
            raise ValueError("n_sample must be a positive integer.")

        self.max_id = max_id
        self.unique_sampling = unique_sampling
        self.max_n_samples = max_n_samples
        self.n_sample = max_n_samples
        if self.unique_sampling:
            self.n_sample = int(self.n_sample * n_samples_multiplier_before_unique)

        with torch.no_grad():
            dist = self.get_log_uniform_distr(max_id, min_id)
            self.register_buffer("dist", dist)
            unique_sampling_dist = self.get_unique_sampling_distr(dist, self.n_sample)
            self.register_buffer("unique_sampling_dist", unique_sampling_dist)

    def get_log_uniform_distr(self, max_id: int, min_id: int = 0) -> torch.Tensor:
        """Approximates the items frequency distribution with log-uniform probability distribution
        with P(class) = (log(class + 2) - log(class + 1)) / log(max_id + 1).
        It assumes item ids are sorted decreasingly by their frequency.

        Parameters
        ----------
        max_id : int
            Maximum discrete value for sampling (e.g. cardinality of the item id)

        Returns
        -------
        torch.Tensor
            Returns the log uniform probability distribution
        """
        log_indices = torch.arange(1.0, max_id - min_id + 2.0, 1.0).log_()
        probs = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
        if min_id > 0:
            probs = torch.cat(
                [torch.zeros([min_id], dtype=probs.dtype), probs], axis=0
            )  # type: ignore
        return probs

    def get_unique_sampling_distr(self, dist, n_sample):
        """Returns the probability that each item is sampled at least once
        given the specified number of trials. This is meant to be used when
        self.unique_sampling == True.
        That probability can be approximated by by 1 - (1 - p)^n
        and we use a numerically stable version: -expm1(num_tries * log1p(-p))
        """
        return (-(-dist.double().log1p_() * n_sample).expm1_()).float()

    def sample(self, labels: torch.Tensor):
        """Sample negative samples and calculate their probabilities.

        If `unique_sampling==True`, then only unique sampled items will be returned.
        The actual # samples will vary from run to run if `unique_sampling==True`,
        as sampling without replacement (`torch.multinomial(..., replacement=False)`) is slow,
        so we use `torch.multinomial(..., replacement=True).unique()`
        which doesn't guarantee the same number of unique sampled items.
        You can try to increase n_samples_multiplier_before_unique
        to increase the chances to have more unique samples in that case.

        Parameters
        ----------
        labels : torch.Tensor, dtype=torch.long, shape=(batch_size,)
            The input labels for which negative samples should be generated.

        Returns
        -------
        neg_samples : torch.Tensor, dtype=torch.long, shape=(n_samples,)
            The unique negative samples drawn from the log-uniform distribution.
        true_probs : torch.Tensor, dtype=torch.float32, shape=(batch_size,)
            The probabilities of the input labels according
            to the log-uniform distribution (depends on self.unique_sampling choice).
        samp_log_probs : torch.Tensor, dtype=torch.float32, shape=(n_samples,)
            The probabilities of the sampled negatives according
            to the log-uniform distribution (depends on self.unique_sampling choice).
        """

        if not torch.is_tensor(labels):
            raise TypeError("Labels must be a torch.Tensor.")
        if labels.dtype != torch.long:
            raise ValueError("Labels must be a tensor of dtype long.")
        if labels.dim() > 2 or (labels.dim() == 2 and min(labels.shape) > 1):
            raise ValueError(
                "Labels must be a 1-dimensional tensor or a 2-dimensional tensor"
                "with one of the dimensions equal to 1."
            )
        if labels.size(0) == 0:
            raise ValueError("Labels must not be an empty tensor.")
        if (labels < 0).any() or (labels > self.max_id).any():
            raise ValueError("All label values must be within the range [0, max_id].")

        n_tries = self.n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(
                self.dist, n_tries, replacement=True  # type: ignore
            ).unique()[: self.max_n_samples]

            device = labels.device
            neg_samples = neg_samples.to(device)

            if self.unique_sampling:
                dist = self.unique_sampling_dist
            else:
                dist = self.dist

            true_probs = dist[labels]  # type: ignore
            samples_probs = dist[neg_samples]  # type: ignore

            return neg_samples, true_probs, samples_probs

    def forward(self, labels):
        return self.sample(labels)
