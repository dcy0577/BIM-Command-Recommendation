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

from typing import List, Optional, Tuple, Type, Union

from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import Tags, TagsType

from merlin_standard_lib import Schema

from ..block.base import SequentialBlock
from ..block.mlp import MLPBlock
from ..tabular.base import (
    TABULAR_MODULE_PARAMS_DOCSTRING,
    AsTabular,
    MergeTabular,
    TabularAggregationType,
    TabularModule,
    TabularTransformationType,
)
from ..utils.torch_utils import get_output_sizes_from_schema
from .continuous import ContinuousFeatures
from .embedding import EmbeddingFeatures, PretrainedEmbeddingFeatures, SoftEmbeddingFeatures

TABULAR_FEATURES_PARAMS_DOCSTRING = """
    continuous_module: TabularModule, optional
        Module used to process continuous features.
    categorical_module: TabularModule, optional
        Module used to process categorical features.
    text_embedding_module: TabularModule, optional
        Module used to process text features.
"""


@docstring_parameter(
    tabular_module_parameters=TABULAR_MODULE_PARAMS_DOCSTRING,
    tabular_features_parameters=TABULAR_FEATURES_PARAMS_DOCSTRING,
)
class TabularFeatures(MergeTabular):
    """Input module that combines different types of features: continuous, categorical & text.

    Parameters
    ----------
    {tabular_features_parameters}
    {tabular_module_parameters}
    """

    CONTINUOUS_MODULE_CLASS: Type[TabularModule] = ContinuousFeatures
    EMBEDDING_MODULE_CLASS: Type[TabularModule] = EmbeddingFeatures
    SOFT_EMBEDDING_MODULE_CLASS: Type[TabularModule] = SoftEmbeddingFeatures
    PRETRAINED_EMBEDDING_MODULE_CLASS: Type[TabularModule] = PretrainedEmbeddingFeatures

    def __init__(
        self,
        continuous_module: Optional[TabularModule] = None,
        categorical_module: Optional[TabularModule] = None,
        pretrained_embedding_module: Optional[TabularModule] = None,
        pre: Optional[TabularTransformationType] = None,
        post: Optional[TabularTransformationType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        **kwargs,
    ):
        to_merge = {}
        if continuous_module:
            to_merge["continuous_module"] = continuous_module
        if categorical_module:
            to_merge["categorical_module"] = categorical_module
        if pretrained_embedding_module:
            to_merge["pretrained_embedding_module"] = pretrained_embedding_module

        assert to_merge != {}, "Please provide at least one input layer"
        super(TabularFeatures, self).__init__(
            to_merge, pre=pre, post=post, aggregation=aggregation, schema=schema, **kwargs
        )

    def project_continuous_features(
        self, mlp_layers_dims: Union[List[int], int]
    ) -> "TabularFeatures":
        """Combine all concatenated continuous features with stacked MLP layers

        Parameters
        ----------
        mlp_layers_dims : Union[List[int], int]
            The MLP layer dimensions

        Returns
        -------
        TabularFeatures
            Returns the same ``TabularFeatures`` object with the continuous features projected
        """
        if isinstance(mlp_layers_dims, int):
            mlp_layers_dims = [mlp_layers_dims]

        continuous = self.to_merge["continuous_module"]
        continuous.aggregation = "concat"

        continuous = SequentialBlock(
            continuous, MLPBlock(mlp_layers_dims), AsTabular("continuous_projection")
        )

        self.to_merge["continuous_module"] = continuous  # type: ignore

        return self

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
        d_output: Optional[int] = None,
        **kwargs,
    ) -> "TabularFeatures":
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
            Indicates if the  soft one-hot encoding technique must be used to
            represent continuous features, by default False

        Returns
        -------
        TabularFeatures
            Returns ``TabularFeatures`` from a dataset schema
        """
        maybe_continuous_module, maybe_categorical_module, maybe_pretrained_module = (
            None,
            None,
            None,
        )

        if continuous_tags:
            if continuous_soft_embeddings:
                maybe_continuous_module = cls.SOFT_EMBEDDING_MODULE_CLASS.from_schema(
                    schema,
                    tags=continuous_tags,
                    **kwargs,
                )
            else:
                maybe_continuous_module = cls.CONTINUOUS_MODULE_CLASS.from_schema(
                    schema, tags=continuous_tags, **kwargs
                )
        if categorical_tags:
            maybe_categorical_module = cls.EMBEDDING_MODULE_CLASS.from_schema(
                schema, tags=categorical_tags, **kwargs
            )
        if pretrained_embeddings_tags:
            maybe_pretrained_module = cls.PRETRAINED_EMBEDDING_MODULE_CLASS.from_schema(
                schema, tags=pretrained_embeddings_tags, **kwargs
            )

        output = cls(
            continuous_module=maybe_continuous_module,
            categorical_module=maybe_categorical_module,
            pretrained_embedding_module=maybe_pretrained_module,
            aggregation=aggregation,
            d_output=d_output,
            **kwargs,
        )

        if automatic_build and schema:
            output.build(
                get_output_sizes_from_schema(
                    schema,
                    kwargs.get("batch_size", -1),
                    max_sequence_length=max_sequence_length,
                ),
                schema=schema,
            )

        if continuous_projection:
            if not automatic_build:
                raise ValueError(
                    "Continuous feature projection can only be done with automatic_build"
                )
            output = output.project_continuous_features(continuous_projection)

        return output

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.merge_values:
            output_sizes.update(in_layer.forward_output_size(input_size))

        return output_sizes

    @property
    def continuous_module(self) -> Optional[TabularModule]:
        if "continuous_module" in self.to_merge:
            return self.to_merge["continuous_module"]

        return None

    @property
    def categorical_module(self) -> Optional[TabularModule]:
        if "categorical_module" in self.to_merge:
            return self.to_merge["categorical_module"]

        return None

    @property
    def pretrained_module(self) -> Optional[TabularModule]:
        if "pretrained_embedding_module" in self.to_merge:
            return self.to_merge["pretrained_embedding_module"]

        return None
