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

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.models.utils.registry import Registry
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, LoftQConfig
from transformers4rec.torch.model.base import Model

transformer_registry: Registry = Registry("transformers")


TRANSFORMER_CONFIG_PARAMETER_DOCSTRING = """        
        d_model: int
            The  hidden dimension of the transformer layer.
        n_head: int
            The number of attention heads in each transformer layer.
        n_layer: int
            The number of transformer layers to stack.
        total_seq_length: int
            The maximum sequence length.
        hidden_act: str, optional
            The activation function in the hidden layers.
            By default 'gelu'
        initializer_range: float, optional
            The standard deviation of the `truncated_normal_initializer`
            for initializing all transformer's weights parameters.
            By default 0.01
        layer_norm_eps: float, optional
            The epsilon used by the layer normalization layers.
            By default 0.03
        dropout: float, optional
            The dropout probability. By default 0.3
        pad_token: int, optional
            The padding token ID. By default 0
        log_attention_weights: bool, optional
            Whether to log attention weights. By default False
"""


class T4RecConfig:
    """A class responsible for setting the configuration of the transformers class
    from Hugging Face and returning the corresponding T4Rec model.
    """

    # this will be called by the torch.TransformerBlock
    def to_huggingface_torch_model(self):
        """
        Instantiate a Hugging Face transformer model based on
        the configuration parameters of the class.

        Returns
        -------
        transformers.PreTrainedModel
            The Hugging Face transformer model.
        """
        model_cls = transformers.MODEL_MAPPING[self.transformers_config_cls]

        # whether to load the pretrained model
        # by doing so you need to make sure that the model config is the same as the pretrained model
        self.if_pretrained = getattr(self, 'if_pretrained', False)
        # whether to use full precision to train a big llama model
        self.full_precision = getattr(self, 'full_precision', False)
        # whether to use quantization only to train a big llama model
        self.use_quantization_only = getattr(self, 'use_quantization_only', False)
        # whether to use loftq to train a big llama model
        self.use_loftq = getattr(self, 'use_loftq', False)
        # whether to use lora to train a big llama model
        self.use_lora = getattr(self, 'use_lora', False)


        if self.transformers_config_cls.model_type == "bert":
            if self.if_pretrained:
                return model_cls.from_pretrained("bert-base-uncased",
                                                #  "bert-large-uncased",
                                                #  torch_dtype=torch.float16,# half precision
                                                 low_cpu_mem_usage=False,
                                                 config=self,
                                                 ignore_mismatched_sizes=True)
            else:
                return model_cls(self)

        if self.transformers_config_cls.model_type == "t5":
            model_cls = transformers.T5Model
            if self.if_pretrained: # by doing so you need to make sure that the model config is the same as the pretrained model
                return model_cls.from_pretrained("t5-small",
                                            #  torch_dtype=torch.float16,# half precision
                                             low_cpu_mem_usage=False,
                                             config=self)
            else:
                return model_cls(self)
        
        if self.transformers_config_cls.model_type == "mixtral":
            model_cls = transformers.MixtralModel
            if self.if_pretrained:
                return model_cls.from_pretrained("mistralai/Mixtral-8x7B-v0.1",
                                            #  torch_dtype=torch.float16,# half precision
                                             low_cpu_mem_usage=False,
                                             config=self)
            else:
                return model_cls(self)
        
        
        if self.transformers_config_cls.model_type == "llama":
            model_cls = transformers.LlamaModel # this works
            if self.if_pretrained:
                bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True) # 8 bits lora
                bnb_config_4bit = BitsAndBytesConfig(  
                                    load_in_4bit=True, 
                                    bnb_4bit_use_double_quant=False, # whether to use double quantization
                                    bnb_4bit_quant_type="nf4",  
                                    bnb_4bit_compute_dtype=torch.float16) # 4 bits qlora
                if self.full_precision:
                    model = model_cls.from_pretrained(
                            "Llama_weights/Llama-3.2-1B",
                            ignore_mismatched_sizes=True, # ignore the embedding layer of original model
                            config=self,
                            low_cpu_mem_usage=False, # initialize without duplicating the parameters, if ignore_mismatched_sizes=True, then low_cpu_mem_usage=False
                            # device_map= 0, # auto is not good for fine-tuning, load the model to gpu
                            )
                elif self.use_quantization_only:
                    model = model_cls.from_pretrained(
                            "Llama_weights/llama-2-7b-hf-weights",
                            # config=self, # use default config
                            torch_dtype=torch.bfloat16,# half precision
                            low_cpu_mem_usage=True, # initialize without duplicating the parameters
                            device_map= 0, # auto is not good for fine-tuning, load the model to gpu
                            quantization_config=bnb_config_8bit,
                            )
                    # freeze the model
                    model = prepare_model_for_kbit_training(model)
                
                # TODO somehow the loftq is not working
                elif self.use_loftq:         
                    model = model_cls.from_pretrained(
                            "Llama_weights/llama-2-7b-chat-hf-weights",
                            config=self,
                            torch_dtype=torch.bfloat16,# half precision
                            low_cpu_mem_usage=True,
                            quantization_config = bnb_config_4bit) # initialize without duplicating the parameters
                
                elif self.use_lora:
                    # try lora
                    model = model_cls.from_pretrained(
                        # "Llama_weights/Llama-2-7b-hf", 
                        # "Llama_weights/Llama-3.2-1B",
                        "Llama_weights/llama-2-7b-hf-weights", 
                        # "Llama_weights/llama-2-13b-hf-weights",
                        # "Llama_weights/llama-2-7b-chat-hf-weights",
                        # config=self,
                        # torch_dtype=torch.bfloat16,# half precision
                        low_cpu_mem_usage=True, # initialize without duplicating the parameters
                        device_map= 0, # auto is not good for fine-tuning, load the model to gpu
                        quantization_config=bnb_config_8bit,) # bnb_config_4bit) # 8 bits lora 
                    model = prepare_model_for_kbit_training(model)

            else:
                model = model_cls(self)
                
            return model
            
                
        return model_cls(self)

    def to_torch_model(
        self,
        input_features,
        *prediction_task,
        task_blocks=None,
        task_weights=None,
        loss_reduction="mean",
        **kwargs
    ):
        """Links the Hugging Face transformer model to the given input block and prediction tasks,
        and returns a T4Rec model.

        Parameters
        ----------
        input_features: torch4rec.TabularSequenceFeatures
            The sequential block that represents the input features and
            defines the masking strategy for training and evaluation.
        prediction_task: torch4rec.PredictionTask
            One or multiple prediction tasks.
        task_blocks: list, optional
            List of task-specific blocks that we apply on top of the HF transformer's output.
        task_weights: list, optional
            List of the weights to use for combining the tasks losses.
        loss_reduction: str, optional
            The reduction to apply to the prediction losses, possible values are:
                'none': no reduction will be applied,
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
            By default: 'mean'.

        Returns
        -------
        torch4rec.Model
            The T4Rec torch model.

        Raises
        ------
        ValueError
            If input block or prediction task is of the wrong type.
        """
        from .. import torch as torch4rec

        if not isinstance(input_features, torch4rec.TabularSequenceFeatures):
            raise ValueError("`input_features` must an instance of SequentialTabularFeatures")
        if not all(isinstance(t, torch4rec.PredictionTask) for t in prediction_task):
            raise ValueError(
                "`task` is of the wrong type, please provide one or multiple "
                "instance(s) of PredictionTask"
            )

        # here in transformerblock we will call the to_huggingface_torch_model
        if self.transformers_config_cls.model_type == "t5":
            body = torch4rec.SequentialBlock(
                input_features, torch4rec.TransformerBlock(self, 
                                                        masking=input_features.masking, 
                                                        embedding_layer=input_features.to_merge.categorical_module.embedding_tables["item_id-list"])
            )
        else:
            body = torch4rec.SequentialBlock(
                input_features, torch4rec.TransformerBlock(self, masking=input_features.masking))

        if self.transformers_config_cls.model_type == "llama" and self.use_lora:
            model = torch4rec.Head(
                body,
                *prediction_task,
                task_blocks=task_blocks,
                task_weights=task_weights,
                loss_reduction=loss_reduction,
            ).to_model(**kwargs)

            print([(n, type(m)) for n, m in model.named_modules()])
            
            # somehow loftq is not working
            # TODO fix the loftq
            # loftq_config = LoftQConfig(loftq_bits=4) # https://huggingface.co/LoftQ/Llama-2-7b-hf-4bit-64rank/tree/main

            config = LoraConfig(
                r=16,
                lora_alpha=64,
                modules_to_save=["heads.0.body.0.projection_module", 
                                 "heads.0.body.0.to_merge.continuous_module.1", 
                                "heads.0.body.0.to_merge.categorical_module.embedding_tables.item_id-list",
                                "heads.0.body.0.to_merge.categorical_module.embedding_tables.classification-list",
                                "heads.0.body.0.to_merge.categorical_module.embedding_tables.target-list",
                                "heads.0.body.0.mh_attention",
                                "heads.0.body.0.encoders",
                                "heads.0.body.0.projection_layer",
                                #  "heads.0.body.0._masking",
                                # "heads.0.body.0.to_merge.pretrained_embedding_module",
                                 "heads.0.prediction_task_dict.next-item.loss", # shall we?
                                 "heads.0.prediction_task_dict.next-item.cat_loss1",
                                 "heads.0.prediction_task_dict.next-item.cat_loss2",
                                 "heads.0.prediction_task_dict.next-item.task_block",
                                 "heads.0.prediction_task_dict.next-item.pre.module",
                                 "heads.0.prediction_task_dict.next-item.mlp_branch1",
                                 "heads.0.prediction_task_dict.next-item.mlp_branch2",],
                target_modules=["q_proj","k_proj","v_proj","o_proj","down_proj","gate_proj","up_proj"], 
                # target_modules=["q_proj","k_proj","v_proj","o_proj",], 
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",)
            
            model = get_peft_model(model, config)
            return model


        return torch4rec.Head(
            body,
            *prediction_task,
            task_blocks=task_blocks,
            task_weights=task_weights,
            loss_reduction=loss_reduction,
        ).to_model(**kwargs)
    
    @property
    def transformers_config_cls(self):
        return self.__class__.__bases__[1]

    @classmethod
    def build(cls, *args, **kwargs):
        raise NotImplementedError


@transformer_registry.register("reformer")
class ReformerConfig(T4RecConfig, transformers.ReformerConfig):
    """Subclass of T4RecConfig and transformers.ReformerConfig from Hugging Face.
    It handles configuration for Reformer layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        axial_pos_shape_first_dim=4,
        **kwargs
    ):
        """
        Creates an instance of ReformerConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}
        axial_pos_shape_first_dim: int, optional
            The first dimension of the axial position encodings.
            During training, the product of the position dims has to be equal to the sequence length.

        Returns
        -------
        ReformerConfig
            An instance of ReformerConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            attention_head_size=d_model,
            attn_layers=["local", "lsh"] * (n_layer // 2) if n_layer > 2 else ["local"],
            num_hidden_layers=n_layer,
            feed_forward_size=d_model * 4,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=dropout,
            lsh_attention_probs_dropout_prob=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            max_position_embeddings=total_seq_length,
            axial_pos_embds_dim=[
                d_model // 2,
                d_model // 2,
            ],
            axial_pos_shape=[
                axial_pos_shape_first_dim,
                total_seq_length // axial_pos_shape_first_dim,
            ],
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("gtp2")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class GPT2Config(T4RecConfig, transformers.GPT2Config):
    """Subclass of T4RecConfig and transformers.GPT2Config from Hugging Face.
    It handles configuration for GPT2 layers in the context of T4Rec models.
    """

    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu_new", # gelu
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of GPT2Config with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        GPT2Config
            An instance of GPT2Config.
        """
        return cls(
            n_embd=d_model,
            n_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            n_positions=total_seq_length,
            n_ctx=total_seq_length,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("longformer")
class LongformerConfig(T4RecConfig, transformers.LongformerConfig):
    """Subclass of T4RecConfig and transformers.LongformerConfig from Hugging Face.
    It handles configuration for LongformerConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of LongformerConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        LongformerConfig
            An instance of LongformerConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            attention_window=total_seq_length,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("electra")
class ElectraConfig(T4RecConfig, transformers.ElectraConfig):
    """Subclass of T4RecConfig and transformers.ElectraConfig from Hugging Face.
    It handles configuration for ElectraConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of ElectraConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        ElectraConfig
            An instance of ElectraConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            embedding_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            intermediate_size=d_model * 4,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=dropout,
            max_position_embeddings=total_seq_length,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("albert")
class AlbertConfig(T4RecConfig, transformers.AlbertConfig):
    """Subclass of T4RecConfig and transformers.AlbertConfig from Hugging Face.
    It handles configuration for AlbertConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of AlbertConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        AlbertConfig
            An instance of AlbertConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_attention_heads=n_head,
            num_hidden_layers=n_layer,
            hidden_act=hidden_act,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=total_seq_length,
            embedding_size=d_model,  # should be same as dimension of the input to ALBERT
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            output_attentions=log_attention_weights,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("xlnet")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class XLNetConfig(T4RecConfig, transformers.XLNetConfig):
    """Subclass of T4RecConfig and transformers.XLNetConfig from Hugging Face.
    It handles configuration for XLNetConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length=None,
        attn_type="bi",
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        mem_len=1,
        **kwargs
    ):
        """
        Creates an instance of XLNetConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}
        mem_len: int,
            The number of tokens to be cached. Pre-computed key/value pairs
            from a previous forward pass are stored and won't be re-computed.
            This parameter is especially useful for long sequence modeling where
            different batches may truncate the entire sequence.
            Tasks like user-aware recommendation could benefit from this feature.
            By default, this parameter is set to 1, which means no caching is used.

        Returns
        -------
        XLNetConfig
            An instance of XLNetConfig.
        """
        return cls(
            d_model=d_model,
            d_inner=d_model * 4,
            n_layer=n_layer,
            n_head=n_head,
            attn_type=attn_type,
            ff_activation=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,
            mem_len=mem_len,
            **kwargs,
        )


@transformer_registry.register("bert")
class BertConfig(T4RecConfig, transformers.BertConfig):
    """Subclass of T4RecConfig and transformers.BertConfig from Hugging Face.
    It handles configuration for BertConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        intermediate_size,
        hidden_act="gelu",
        initializer_range=0.02, #0.01 for t4rec
        layer_norm_eps=1e-12, #0.03 for t4rec
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of BertConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        BertConfig
            An instance of BertConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            intermediate_size = intermediate_size,# for bert-large
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            max_position_embeddings=total_seq_length,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("roberta")
class RobertaConfig(T4RecConfig, transformers.RobertaConfig):
    """Subclass of T4RecConfig and transformers.RobertaConfig from Hugging Face.
    It handles configuration for RobertaConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of RobertaConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        RobertaConfig
            An instance of RobertaConfig.
        """
        # To account for target positions at inference mode, we extend the maximum sequence length.
        total_seq_length = total_seq_length + 2
        return cls(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            max_position_embeddings=total_seq_length,
            vocab_size=1,
            **kwargs,
        )


@transformer_registry.register("transfo-xl")
class TransfoXLConfig(T4RecConfig, transformers.TransfoXLConfig):
    """Subclass of T4RecConfig and transformers. TransfoXLConfig from Hugging Face.
    It handles configuration for TransfoXLConfig layers in the context of T4Rec models.
    """

    @docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
    @classmethod
    def build(
        cls,
        d_model,
        n_head,
        n_layer,
        total_seq_length,
        hidden_act="gelu",
        initializer_range=0.01,
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of TransfoXLConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        TransfoXLConfig
            An instance of TransfoXLConfig.
        """
        return cls(
            d_model=d_model,
            d_embed=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_inner=d_model * 4,
            hidden_act=hidden_act,
            untie_r=True,
            attn_type=0,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            pad_token_id=pad_token,
            output_attentions=log_attention_weights,
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
            mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
            div_val=1,  # Disables adaptative input (embeddings), because the embeddings are managed by TabularFeatures
            **kwargs,
        )

@transformer_registry.register("llama")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class LlamaConfig(T4RecConfig, transformers.LlamaConfig):
    """Subclass of T4RecConfig and transformers.llamaConfig from Hugging Face.
    It handles configuration for llama layers in the context of T4Rec models.
    """

    @classmethod
    def build(
        cls,
        hidden_size = 2048,
        intermediate_size = 8192, #11008 for 7b, 8192 for 1b, 13824 for 13b
        n_head = 32,
        n_layer = 32,
        total_seq_length = 2048,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        pad_token=0,
        **kwargs
    ):
        """
        Creates an instance of llamaConfig with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        llamaConfig
            An instance of llamaConfig.
        """
        # llama2.7b
        # return cls(
        #     hidden_size=hidden_size, # 4096
        #     # intermediate_size=hidden_size * 4, # 11008
        #     intermediate_size= intermediate_size, # have to match the pretrained weights
        #     num_hidden_layers=n_layer, 
        #     num_attention_heads=n_head,
        #     num_key_value_heads=n_head,
        #     hidden_act=hidden_act,
        #     initializer_range=initializer_range, #0.02
        #     max_position_embeddings=total_seq_length,
        #     # vocab_size=32000, # just to save the memory
        #     vocab_size=1, # just to save the memory
        #     rms_norm_eps=rms_norm_eps, # 1e-5
        #     pad_token_id=pad_token,
        #     **kwargs,
        # )

        # llama3.1b
        # load parameter dict from the json file
        import json
        with open('Llama_weights/Llama-3.2-1B/config.json') as f:
            data = json.load(f)
        # save some memory
        data["vocab_size"] = 1
        data["num_hidden_layers"] = n_layer
        data["num_attention_heads"] = n_head
        data["hidden_size"] = hidden_size
        data["intermediate_size"] = intermediate_size
        data["max_position_embeddings"] = total_seq_length
        data["output_attentions"]=True,
        data.update(kwargs)

        return cls(**data)

@transformer_registry.register("t5")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class T5Config(T4RecConfig, transformers.T5Config):
    """Subclass of T4RecConfig and transformers.T5Config from Hugging Face.
    It handles configuration for T5 layers in the context of T4Rec models.
    """

    @classmethod
    def build(
        cls,
        d_model=768,
        n_head=12,
        n_layer=12,
        total_seq_length= 512,
        initializer_range=1, # 0.01
        layer_norm_eps=0.03,
        dropout=0.3,
        pad_token=0,
        log_attention_weights=False,
        **kwargs
    ):
        """
        Creates an instance of T5Config with the given parameters.

        Parameters
        ----------
        {transformer_cfg_parameters}

        Returns
        -------
        T5Config
            An instance of T5Config.
        """
        return cls(
            d_model=d_model, # 512 \ 768
            d_ff=2048, # 2048
            num_layers=n_layer, # 6
            num_heads=n_head, # 8
            initializer_range=initializer_range,
            # resid_pdrop=dropout,
            # embd_pdrop=dropout,
            # attn_pdrop=dropout,
            # n_positions=total_seq_length,
            # n_ctx=total_seq_length,
            # max_position_embeddings=total_seq_length,
            # output_attentions=log_attention_weights,
            vocab_size=1,
            pad_token_id=pad_token,
            **kwargs,
        )

transformer_registry.register("mixtral")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class MixtralConfig(T4RecConfig, transformers.MixtralConfig):
    """Subclass of T4RecConfig. It handles configuration for mixtral layers in the context of T4Rec models.
    """
    @classmethod
    def build(
        cls,
        hidden_size = 4096,
        intermediate_size = 14336,
        n_head = 32,
        n_layer = 32,
        num_experts_per_tok=2,
        num_local_experts=8,
        total_seq_length = 4096 * 8,
        hidden_act="silu",
        pad_token=0,
        **kwargs
    ):

        return cls(
            vocab_size=1, # we dont need embedding layer
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=8,
            num_experts_per_tok=num_experts_per_tok,
            num_local_experts=num_local_experts,
            hidden_act="silu",
            max_position_embeddings=total_seq_length, # default for 8x7b 4096 * 32, for 4*7b 4096 * 8
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=pad_token,
            bos_token_id=0,
            eos_token_id=0,
            tie_word_embeddings=False,
            rope_theta=1e6,
            sliding_window=None,
            attention_dropout=0.0,
            output_router_logits=False, # for the auxillary loss
            router_aux_loss_coef=0.001,
            output_attentions=False,
            **kwargs,
        )

transformer_registry.register("mistral")
@docstring_parameter(transformer_cfg_parameters=TRANSFORMER_CONFIG_PARAMETER_DOCSTRING)
class MistralConfig(T4RecConfig, transformers.MistralConfig):
    """Subclass of T4RecConfig. It handles configuration for mistral layers in the context of T4Rec models.
    """
    @classmethod
    def build(
        cls,
        hidden_size = 4096,
        intermediate_size = 14336,
        n_head = 32,
        n_layer = 32,
        total_seq_length = 4096 * 32,
        hidden_act="silu",
        pad_token=0,
        **kwargs
    ):

        return cls(
            vocab_size=1, # we dont need embedding layer
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=total_seq_length, # default for 8x7b 4096 * 32, for 4*7b 4096 * 8
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=pad_token,
            bos_token_id=0,
            eos_token_id=0,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            sliding_window=4096,
            attention_dropout=0.0,
            **kwargs,
        )