import torch
import torch.nn as nn
# [MODIFIED]
# from transformers import Olmo2ForCausalLM
from transformers.models.olmo2 import Olmo2ForCausalLM

from transformers import PretrainedConfig
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, PreTrainedModel
import torch

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """
    # [MODIFIED]
    model_type = "medusa"

    def __init__(
        self,
        medusa_num_heads: int | None = None,
        medusa_num_layers: int | None = None,
        base_model_name_or_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU() # 

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x)) 


class MedusaModelABC(PreTrainedModel):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers), 
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        kwargs.pop("config", None)
        try:
            print("MedusaModelBC from pretrianed")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            # print("config.medusa_num_heads = ", config.medusa_num_heads)
            # config.medusa_num_heads = 1 # TODO
            print("from pretrained: medusa_num_heads = ", config.medusa_num_heads)
            
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            print("except")
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = config.medusa_num_heads # TODO: fix the uploaded config (only include 2 heads)
            # base_model_config.medusa_num_heads = 5  # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            print("from pretrained: medusa_num_heads = ", config.medusa_num_heads)
            return model
        
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
    
class MedusaModelOlmo(MedusaModelABC, Olmo2ForCausalLM):
    config_class = MedusaConfig
    pass

class MedusaModel():
    config_class = MedusaConfig
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        print("from_pretrained")
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            print("here")
        except Exception:
            config = None

        # non-medusa configs: treat as plain olmo2
        return MedusaModelOlmo.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )

def print_dynamic_cache_shapes(past_key_values):
    """
    Print the shape of all key/value tensors in a DynamicCache object.
    
    Args:
        past_key_values (DynamicCache): Returned from model(..., use_cache=True)
    """
    for i, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
        print(f"past_key_values: Layer {i} - key shape: {tuple(k.shape)}, value shape: {tuple(v.shape)}")
        break
