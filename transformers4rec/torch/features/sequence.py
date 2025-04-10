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

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Tags, TagsType

from merlin_standard_lib import Schema

from ..block.base import BlockOrModule, BuildableBlock, SequentialBlock
from ..block.mlp import MLPBlock
from ..masking import MaskSequence, masking_registry
from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    AsTabular,
    TabularAggregationType,
    TabularModule,
    TabularTransformationType,
)
from . import embedding
from .tabular import TABULAR_FEATURES_PARAMS_DOCSTRING, TabularFeatures


def create_last_item_mask(sequence):
    mask = torch.zeros_like(sequence, dtype=torch.bool)
    for i in range(sequence.size(0)):
        seq = sequence[i]
        non_padding_indices = (seq != 0).nonzero(as_tuple=False).squeeze()
        if len(non_padding_indices) > 0:
            last_idx = non_padding_indices[-1]
            mask[i, last_idx] = True
    return mask

@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    embedding_features_parameters=embedding.EMBEDDING_FEATURES_PARAMS_DOCSTRING,
)
class SequenceEmbeddingFeatures(embedding.EmbeddingFeatures):
    """Input block for embedding-lookups for categorical features. This module produces 3-D tensors,
    this is useful for sequential models like transformers.

    Parameters
    ----------
    {embedding_features_parameters}
    padding_idx: int
        The symbol to use for padding.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        feature_config: Dict[str, embedding.FeatureConfig],
        item_id: Optional[str] = None,
        padding_idx: int = 0,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
    ):
        self.padding_idx = padding_idx
        super(SequenceEmbeddingFeatures, self).__init__(
            feature_config=feature_config,
            item_id=item_id,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
        )

    def table_to_embedding_module(self, table: embedding.TableConfig) -> torch.nn.Embedding:
        embedding_table = torch.nn.Embedding(
            table.vocabulary_size, table.dim, padding_idx=self.padding_idx
        )
        if table.initializer is not None:
            table.initializer(embedding_table.weight)
        return embedding_table

    def forward_output_size(self, input_sizes):
        sizes = {}

        for fname, fconfig in self.feature_config.items():
            fshape = input_sizes[fname]
            sizes[fname] = torch.Size(list(fshape) + [fconfig.table.dim])

        return sizes


class FusionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(1024, 8)
        self.norm = torch.nn.LayerNorm(1024)
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
class TabularSequenceFeatures(TabularFeatures):
    """Input module that combines different types of features to a sequence: continuous,
    categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    projection_module: BlockOrModule, optional
        Module that's used to project the output of this module, typically done by an MLPBlock.
    masking: MaskSequence, optional
         Masking to apply to the inputs.
    {tabular_module_parameters}

    """

    EMBEDDING_MODULE_CLASS = SequenceEmbeddingFeatures

    def __init__(
        self,
        continuous_module: Optional[TabularModule] = None,
        categorical_module: Optional[TabularModule] = None,
        pretrained_embedding_module: Optional[TabularModule] = None,
        projection_module: Optional[BlockOrModule] = None,
        masking: Optional[MaskSequence] = None,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        d_output: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            continuous_module,
            categorical_module,
            pretrained_embedding_module,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            **kwargs
        )
        self.projection_module = projection_module
        self.set_masking(masking)
        # easier to set!!
        self.id_only_baseline = kwargs.get("id_only_baseline", False)
        if self.id_only_baseline:
            self.multi_task_labels = False
            self.custom_aggregation = False
            self.cross_attention_agg = False
            self.self_attention_agg = False
            self.concatenation = False
            self.mlp_pooling = False
            self.att_pooling = False
        else:
            # lets try multi task prediction using multi categorical labels
            self.multi_task_labels = kwargs.get("multi_task_labels", False)
            self.custom_aggregation = kwargs.get("custom_aggregation", False)
            self.cross_attention_agg = kwargs.get("cross_attention_agg", False)
            self.self_attention_agg = kwargs.get("self_attention_agg", False)
            self.concatenation = kwargs.get("concatenation", False)
            self.mlp_pooling = kwargs.get("mlp_pooling", False)
            self.att_pooling = kwargs.get("att_pooling", False)

        # -----------------llm augmented multi task-----------------
        self.additional_category_label_names = ["classification-list", "target-list"]
        # -----------------llm augmented multi task-----------------

        # self.additional_category_label_names = ["cat-list", "merge_count_norm-list"]

        # lets try to fuse the features using transformer
        # self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=1024, nhead=4, dim_feedforward=2048, batch_first=True)
        if self.custom_aggregation:
            # this is for mapping the features to the same dimension
            self.feature_dims = {"item_id-list":1024,  # 512 for t5
                            # "cat-list": 64, 
                            "classification-list": 1024, 
                            "target-list": 1024, 
                            "continuous_projection": 1024,
                            "pretrained_item_id_embeddings": 3072, #3072
                            } # for cat, classification, target, contius_projection, pretrained_embedding
                
            if self.self_attention_agg:
                self.mh_attention = torch.nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True)
                # self.encoders = torch.nn.ModuleDict(
                #     {k: torch.nn.Linear(v, 1024) for k, v in self.feature_dims.items()}
                # )
                self.encoders = torch.nn.ModuleDict({
                    k: torch.nn.Sequential(
                        torch.nn.Linear(v, 1024),
                        torch.nn.LayerNorm(1024),  # norm before activation
                        # torch.nn.ReLU()
                    ) for k, v in self.feature_dims.items()
                })
                if self.mlp_pooling:
                    self.mlp_layer = torch.nn.Linear(1024*len(self.feature_dims), d_output) # bert-base
                    # self.relu = torch.nn.ReLU(inplace=True)
                    # self.projection_layer = torch.nn.Linear(1024, 2048, bias=True) # llama: 1024 -> 2048
                    # self.projection_layer = torch.nn.Linear(1024, 1024, bias=True) # mixtral 1024 -> 1536
                    # self.projection_layer = torch.nn.Linear(1024, 512, bias=True) # t5 1024 -> 512
                if self.att_pooling:
                    self.query = torch.nn.Parameter(torch.randn(1,1024))
                    # self.projection_layer = torch.nn.Linear(1024, 2048, bias=True) # llama: 1024 -> 2048
                    self.projection_layer = torch.nn.Linear(1024, d_output, bias=True) # mixtral 1024 -> 1536
                    # self.projection_layer = torch.nn.Linear(1024, 512, bias=True) # t5 1024 -> 512

            elif self.cross_attention_agg:
                # lets try to fuse the features using cross attention
                self.cross_attention = torch.nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
                self.linear_key = torch.nn.Linear(1024, 1024)
                self.linear_value = torch.nn.Linear(1024, 1024)
                self.encoders = torch.nn.ModuleDict(
                    {k: torch.nn.Linear(v, 1024) for k, v in self.feature_dims.items()}
                )
                self.layer_norm_fusion = torch.nn.LayerNorm(1024)
                self.projection_layer = torch.nn.Linear(1024, d_output, bias=True)
            elif self.concatenation:
                self.encoders = torch.nn.ModuleDict(
                    {k: torch.nn.Linear(v, 1024) for k, v in self.feature_dims.items()}
                )
                # 1024*6 -> 6144 -> d_model
                # 1024*5 -> 5120 -> d_model
                self.projection_layer = torch.nn.Linear(5120, d_output, bias=True)
            
    @classmethod
    def from_schema(  # type: ignore
        cls,
        schema: Schema,
        continuous_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CONTINUOUS,),
        categorical_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.CATEGORICAL,),
        pretrained_embeddings_tags: Optional[Union[TagsType, Tuple[Tags]]] = (Tags.EMBEDDING,),
        aggregation: Optional[str] = None,
        automatic_build: bool = True,
        max_sequence_length: Optional[int] = None,
        continuous_projection: Optional[Union[List[int], int]] = None,
        continuous_soft_embeddings: bool = False,
        projection: Optional[Union[torch.nn.Module, BuildableBlock]] = None,
        d_output: Optional[int] = None,
        masking: Optional[Union[str, MaskSequence]] = None,
        **kwargs
    ) -> "TabularSequenceFeatures":
        """Instantiates ``TabularFeatures`` from a ``DatasetSchema``

        Parameters
        ----------
        schema : DatasetSchema
            Dataset schema
        continuous_tags : Optional[Union[TagsType, Tuple[Tags]]], optional
            Tags to filter the continuous features, by default Tags.CONTINUOUS
        categorical_tags : Optional[Union[TagsType, Tuple[Tags]]], optional
            Tags to filter the categorical features, by default Tags.CATEGORICAL
        aggregation : Optional[str], optional
            Feature aggregation option, by default None
        automatic_build : bool, optional
            Automatically infers input size from features, by default True
        max_sequence_length : Optional[int], optional
            Maximum sequence length for list features by default None
        continuous_projection : Optional[Union[List[int], int]], optional
            If set, concatenate all numerical features and project them by a number of MLP layers.
            The argument accepts a list with the dimensions of the MLP layers, by default None
        continuous_soft_embeddings : bool
            Indicates if the  soft one-hot encoding technique must be used to represent
            continuous features, by default False
        projection: Optional[Union[torch.nn.Module, BuildableBlock]], optional
            If set, project the aggregated embeddings vectors into hidden dimension vector space,
            by default None
        d_output: Optional[int], optional
            If set, init a MLPBlock as projection module to project embeddings vectors,
            by default None
        masking: Optional[Union[str, MaskSequence]], optional
            If set, Apply masking to the input embeddings and compute masked labels, It requires
            a categorical_module including an item_id column, by default None

        Returns
        -------
        TabularFeatures
            Returns ``TabularFeatures`` from a dataset schema
        """
        output: TabularSequenceFeatures = super().from_schema(  # type: ignore
            schema=schema,
            continuous_tags=continuous_tags,
            categorical_tags=categorical_tags,
            pretrained_embeddings_tags=pretrained_embeddings_tags,
            aggregation=aggregation,
            automatic_build=automatic_build,
            max_sequence_length=max_sequence_length,
            continuous_projection=continuous_projection,
            continuous_soft_embeddings=continuous_soft_embeddings,
            d_output=d_output,
            **kwargs
        )
        if d_output and projection:
            raise ValueError("You cannot specify both d_output and projection at the same time")
        if (projection or masking or d_output) and not aggregation:
            # TODO: print warning here for clarity
            output.aggregation = "concat"  # type: ignore
        hidden_size = output.output_size()

        if d_output and not projection:
            projection = MLPBlock([d_output])
        if projection and hasattr(projection, "build"):
            projection = projection.build(hidden_size)  # type: ignore
        if projection:
            output.projection_module = projection  # type: ignore
            hidden_size = projection.output_size()  # type: ignore

        if isinstance(masking, str):
            masking = masking_registry.parse(masking)(
                hidden_size=output.output_size()[-1], **kwargs
            )
        if masking and not getattr(output, "item_id", None):
            raise ValueError("For masking a categorical_module is required including an item_id.")
        output.set_masking(masking)  # type: ignore

        return output

    @property
    def masking(self):
        return self._masking

    def set_masking(self, value):
        self._masking = value

    @property
    def item_id(self) -> Optional[str]:
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_id", None)

        return None

    @property
    def item_embedding_table(self) -> Optional[torch.nn.Module]:
        if "categorical_module" in self.to_merge:
            return getattr(self.to_merge["categorical_module"], "item_embedding_table", None)

        return None

    def forward(self, inputs, training=False, testing=False, **kwargs):
        outputs = super(TabularSequenceFeatures, self).forward(inputs)
        # check if the key is in the dict
        if "pretrained_item_id_embeddings" in inputs:
            outputs["pretrained_item_id_embeddings"] = inputs["pretrained_item_id_embeddings"]
        if self.custom_aggregation:
            # this is element level self attention
            if self.self_attention_agg:
                # we want to modeling the feature on the item level
                feature_list = []
                for k, v in outputs.items():
                        if k in self.feature_dims:
                            x = self.encoders[k](v) # (batch_size, seq_length, 1024)
                            feature_list.append(x)

                stack_feature = torch.stack(feature_list, dim=2) # (batch_size, seq_length, num_features, feature_dim)
                batch_size, seq_len, num_features, feature_dim = stack_feature.size()
                # merge the features on the item level
                # vectorized the attention computation instead of for loop
                stack_feature = stack_feature.permute(1, 0, 2, 3)  # [seq_len, batch_size, num_features, feature_dim]
                attn_input = stack_feature.reshape(-1, num_features, feature_dim) # [seq_len*batch_size, num_features, feature_dim]
                fused_feature, att_weights = self.mh_attention(attn_input, attn_input, attn_input) # [seq_len*batch_size, num_features, feature_dim]

                if self.att_pooling:
                    # attention pooling
                    # query = self.query_layer(fused_feature.mean(dim=1)) # [seq_len*batch_size, feature_dim]
                    query = self.query.expand(fused_feature.size(0), -1) # [seq_len*batch_size, feature_dim]
                    attn_scores = torch.bmm(fused_feature, query.unsqueeze(2)).squeeze(-1)/ math.sqrt(feature_dim) # [seq_len*batch_size, num_features]
                    attn_weights = F.softmax(attn_scores, dim=1)
                    output = (fused_feature * attn_weights.unsqueeze(-1)).sum(dim=1) # [seq_len*batch_size, feature_dim]
                    output = output.reshape(seq_len, batch_size, -1).permute(1, 0, 2) # [batch_size, seq_len, feature_dim]
                    outputs = self.projection_layer(output) # 1024 -> [batch_size, seq_len, 2048]

                # MLP pooling
                if self.mlp_pooling:
                    # [seq_len*batch_size, num_features, feature_dim] -> [batch_size, seq_len, num_features*feature_dim]
                    outputs = fused_feature.reshape(seq_len, batch_size, -1).transpose(0, 1).contiguous().flatten(2)
                    outputs = self.mlp_layer(outputs) # num_features*feature_dim -> [batch_size, seq_len, 2048]
            
            # this is element level cross attention using item id
            elif self.cross_attention_agg:
                feature_list = []
                for k, v in outputs.items():
                        if k in self.feature_dims:
                            x = self.encoders[k](v) # (batch_size, seq_length, 1024)
                            feature_list.append(x)
                item_id_query = feature_list[0] # (batch_size, seq_length, 1024)
                cat_feature_list_keys = torch.cat(feature_list[1:], dim=1) # (batch_size, seq_length*num_features, 1024)
                cat_feature_list_values = torch.cat(feature_list[1:], dim=1) # (batch_size, seq_length*num_features, 1024)
                k = self.linear_key(cat_feature_list_keys) # (batch_size, seq_length*num_features, 1024)
                v = self.linear_value(cat_feature_list_values) # (batch_size, seq_length*num_features, 1024)
                fused_feature, _ = self.cross_attention(item_id_query,k,v)  # (batch_size, seq_length, 1024)
                # residual connection, original approach in transformer
                fused_feature = self.layer_norm_fusion(fused_feature + item_id_query)

                # project the fused feature to the same dimension
                outputs = self.projection_layer(fused_feature)  # 1024 -> [batch_size, seq_len, 2048]

            elif self.concatenation:
                feature_list = []
                for k, v in outputs.items():
                        if k in self.feature_dims:
                            x = self.encoders[k](v)
                            feature_list.append(x)
                outputs = torch.cat(feature_list, dim=-1)
                outputs = self.projection_layer(outputs)
        

        # This is sequence level cross attention using item_id
        # elif self.multi_task_labels:
        #     #--------------------------new--------------------------
        #     outputs_item_ids = outputs['item_id-list'] # (batch_size, seq_length, 1024)

        #     outputs_cat = outputs['cat-list'] # (batch_size, seq_length, 64)
        #     outputs_classification = outputs['classification-list'] # (batch_size, seq_length, 64)
        #     outputs_target = outputs['target-list'] # (batch_size, seq_length, 64)
        #     outputs_merge_count = outputs['continuous_projection'] # (batch_size, seq_length, 64)
        #     outputs_pretrained_embedding = outputs['pretrained_item_id_embeddings'] # (batch_size, seq_length, 1024)

            # # the following is modeling the feature on sequence level
            # outputs_merge = torch.cat([outputs_cat, outputs_classification, outputs_target, outputs_merge_count, outputs_pretrained_embedding], dim=-1) # (batch_size, seq_length, 1472)
            # # q = self.cross_attention_proj_item_id(outputs_item_ids) # (batch_size, seq_length, 1024)
            # q = outputs_item_ids
            # k = self.cross_attention_proj_k(outputs_merge) # (batch_size, seq_length, 1024)
            # v = self.cross_attention_proj_v(outputs_merge) # (batch_size, seq_length, 1024)
            # padding_mask = (inputs['item_id-list'] == 0)  # (batch_size, seq_length)
            # last_item_mask = create_last_item_mask(inputs['item_id-list']) # (batch_size, seq_length)
            # key_padding_mask = padding_mask | last_item_mask  # (batch_size, seq_length)
            # outputs, _ = self.cross_attention(q, k, v, key_padding_mask=key_padding_mask) # (batch_size, seq_length, 1024)
            #--------------------------new--------------------------

        else:
            # for attention fusion based agregation we dont do aggregation
            if self.masking or self.projection_module:
                outputs = self.aggregation(outputs)

            if self.projection_module:
                outputs = self.projection_module(outputs)
                # the direct use might disclose the futrue information

                # seq_len = outputs.size(1)
                # causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(outputs.device)
                # outputs = self.encoder_layer(outputs, src_mask=causal_mask)

        if self.masking:
            if self.multi_task_labels:
                outputs = self.masking(
                    outputs,
                    item_ids=self.to_merge["categorical_module"].item_seq,
                    training=training,
                    testing=testing,
                    additional_cat_ids={k: inputs[k] for k in self.additional_category_label_names if k in inputs},
                )
            else:
                outputs = self.masking(
                    outputs,
                    item_ids=self.to_merge["categorical_module"].item_seq,
                    training=training,
                    testing=testing,
                )

        return outputs

    def project_continuous_features(self, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions]

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "concat"

        continuous = SequentialBlock(
            continuous, MLPBlock(dimensions), AsTabular("continuous_projection")
        )

        self.to_merge["continuous_module"] = continuous

        return self

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.merge_values:
            output_sizes.update(in_layer.forward_output_size(input_size))

        output_sizes = self._check_post_output_size(output_sizes)

        if self.projection_module:
            output_sizes = self.projection_module.output_size()

        return output_sizes


TabularFeaturesType = Union[TabularSequenceFeatures, TabularFeatures]
