import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.olmo2 import Olmo2ForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class MedusaCausalLMOutput(CausalLMOutputWithPast):
    medusa_logits: Optional[List[torch.Tensor]] = None

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """
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
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
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
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = config.medusa_num_heads # TODO: fix the uploaded config (only include 2 heads)
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
            # for i in range(model.medusa):
            #     model.medusa_head[i][-1].weight.data[:] = model.lm_head.weight.data[:]
            #     print(f"copy lm_head to medusa head {i}")
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through the base model
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)
        
        medusa_logits = []
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
            
        if not return_dict:
            return (logits,) + outputs[1:] + (medusa_logits,)
            
        return MedusaCausalLMOutput(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            medusa_logits=medusa_logits
        )

    def base_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Forward pass of the base model only (no Medusa heads).
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through the base model
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)
        
        if not return_dict:
            return (logits,) + outputs[1:]
            
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def medusa_generate(
        self,
        input_ids: torch.LongTensor,
        max_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        gamma: int = 4,
        eos_token_id: int = None,
    ):
        """
        Optimized generation loop for Medusa models.
        Fuses drafting and verification to ensure 1 backbone pass per step.
        """
        from sampling.utils import norm_logits, sample
        # print(f"Max_len = {max_len}")
        # 1. Prefill
        # Run the model on the input_ids to get the initial hidden states and KV cache
        outputs = self.base_model.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs[0] # (batch, seq_len, hidden_size)
        
        # The last hidden state is used to draft the next tokens
        last_hidden_state = hidden_states[:, -1:, :]
        
        cur_len = input_ids.shape[1]
        final_input_ids = input_ids.clone()
        all_scores = []
        
        step = 0
        step_count = 0
        while final_input_ids.shape[1] < max_len:
            step_count += 1
            # 2. Draft
            # Use medusa heads to generate gamma draft tokens
            # last_hidden_state: (batch, 1, hidden_size)
            
            draft_tokens = []
            draft_probs = []
            
            curr_hidden = last_hidden_state
            
            # First draft token comes from the Base Model (lm_head)
            base_logits = self.base_model.lm_head(curr_hidden) # (batch, 1, vocab)
            base_probs = norm_logits(base_logits[:, -1, :], temperature, top_k, top_p)
            token_0 = sample(base_probs)
            draft_tokens.append(token_0)
            all_scores.append(base_logits)
            
            # Subsequent draft tokens come from Medusa heads
            medusa_logits_list = []
            for i in range(self.medusa):
                # head[i] predicts t+i+2 (relative to curr_hidden's t)
                medusa_logits = self.medusa_head[i](curr_hidden) # (batch, 1, vocab)
                medusa_logits_list.append(medusa_logits)
            
            # Now sample from these logits
            # We need gamma tokens total. We already have 1 (token_0).
            # So we need gamma-1 more.
            
            for i in range(gamma - 1):
                if i < len(medusa_logits_list):
                    logits = medusa_logits_list[i]
                    probs = norm_logits(logits[:, -1, :], temperature, top_k, top_p)
                    token = sample(probs)
                    draft_tokens.append(token)
                    draft_probs.append(probs)
                else:
                    break
            
            draft_input_ids = torch.cat(draft_tokens, dim=1) # (batch, gamma)
            
            # 3. Verify
            # Run base model on the draft tokens
            target_outputs = self.base_model.model(
                input_ids=draft_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            target_hidden_states = target_outputs[0] # (batch, gamma, hidden_size)
            target_logits = self.base_model.lm_head(target_hidden_states) # (batch, gamma, vocab)
            new_past_key_values = target_outputs.past_key_values
            
            # Rejection Sampling
            # We verify draft_tokens[1:] against target_logits
            # draft_tokens[0] is accepted.
            # target_logits[0] verifies draft_tokens[1]
            # target_logits[1] verifies draft_tokens[2]
            
            accepted_count = 1 # token_0 is always accepted
            
            # Check draft_tokens[1:]
            # There are len(draft_tokens)-1 candidates to verify
            
            for i in range(len(draft_tokens) - 1):
                # We are verifying draft_tokens[i+1]
                # Using target_logits[i]
                
                token_id = draft_tokens[i+1].item()
                target_probs = norm_logits(target_logits[:, i, :], temperature, top_k, top_p)
                draft_prob = draft_probs[i][0, token_id].item()
                target_prob = target_probs[0, token_id].item()
                
                if torch.rand(1).item() < min(1.0, target_prob / draft_prob):
                    accepted_count += 1
                else:
                    # Rejected draft_tokens[i+1]
                    # Resample from target_probs (which corresponds to position of draft_tokens[i+1])
                    resampled_token = sample(target_probs)
                    
                    # Correction
                    # We keep draft_tokens[:accepted_count] and append resampled_token
                    # accepted_count includes token_0 and any accepted medusa tokens
                    
                    final_input_ids = torch.cat([final_input_ids, draft_input_ids[:, :accepted_count], resampled_token], dim=1)
                    
                    # Collect scores
                    # We accepted accepted_count tokens (including token_0).
                    # token_0 scores are already in all_scores.
                    # We need scores for the accepted medusa tokens + resampled token.
                    # Medusa tokens: draft_tokens[1:accepted_count] -> target_logits[:, 0:accepted_count-1]
                    # Resampled token: target_logits[:, accepted_count-1] (which is target_logits[:, i])
                    # So we want target_logits[:, 0:accepted_count]
                    
                    all_scores.append(target_logits[:, :accepted_count, :])
                    
                    # Update KV cache
                    # We need to keep KV for accepted_count tokens.
                    # new_past_key_values has KV for all draft tokens.
                    # We slice it to accepted_count.
                    
                    # Helper to slice KV cache
                    def slice_kv(past, length):
                        if past is None: return None
                        
                        # Handle DynamicCache
                        if hasattr(past, 'key_cache'): 
                            # Modify in-place
                            for i in range(len(past.key_cache)):
                                past.key_cache[i] = past.key_cache[i][..., :length, :]
                                past.value_cache[i] = past.value_cache[i][..., :length, :]
                            
                            if hasattr(past, '_seen_tokens'):
                                past._seen_tokens = length
                            elif hasattr(past, 'seen_tokens'):
                                past.seen_tokens = length
                            return past
                        
                        # Handle Tuple
                        new_past = []
                        for layer in past:
                            k, v = layer
                            new_k = k[..., :length, :]
                            new_v = v[..., :length, :]
                            new_past.append((new_k, new_v))
                        return tuple(new_past)
                    
                    # The total length in new_past_key_values is cur_len + gamma
                    # We want to keep cur_len + accepted_count
                    keep_len = cur_len + accepted_count
                    past_key_values = slice_kv(new_past_key_values, keep_len)
                    
                    # Run forward on correction token
                    correction_outputs = self.base_model.model(
                        input_ids=resampled_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = correction_outputs.past_key_values
                    last_hidden_state = correction_outputs[0] # (batch, 1, hidden_size)
                    
                    step += (accepted_count + 1)
                    cur_len += (accepted_count + 1)
                    break
            
            else:
                # All accepted
                accepted_count = len(draft_tokens)
                final_input_ids = torch.cat([final_input_ids, draft_input_ids], dim=1)
                
                # Sample one more token
                # target_logits[accepted_count - 1] predicts the token AFTER the last draft token
                last_logits = target_logits[:, accepted_count - 1, :]
                last_probs = norm_logits(last_logits, temperature, top_k, top_p)
                extra_token = sample(last_probs)
                
                final_input_ids = torch.cat([final_input_ids, extra_token], dim=1)
                
                # Collect scores
                # We accepted all drafts (excluding token_0, that's accepted_count-1 tokens) + 1 extra
                # Total added from target_logits: accepted_count
                # target_logits has shape (batch, gamma, vocab).
                # We want target_logits[:, :accepted_count, :]
                all_scores.append(target_logits[:, :accepted_count, :])
                
                # Update KV cache
                past_key_values = new_past_key_values
                
                extra_outputs = self.base_model.model(
                    input_ids=extra_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = extra_outputs.past_key_values
                last_hidden_state = extra_outputs[0]
                
                step += (accepted_count + 1)
                cur_len += (accepted_count + 1)

            if eos_token_id is not None and (final_input_ids[:, input_ids.shape[1]:] == eos_token_id).any():
                # print("eos: ", final_input_ids[:, input_ids.shape[1]:])
                # Truncate at first EOS
                new_tokens = final_input_ids[:, input_ids.shape[1]:]
                
                eos_idx = (new_tokens[0] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_idx) > 0:
                    first_eos = eos_idx[0].item()
                    final_input_ids = final_input_ids[:, :input_ids.shape[1] + first_eos + 1]
                break
            
            # if final_input_ids.shape[1] >= max_len:
            #     print(">= max_len")
                
        # Concatenate scores
        if all_scores:
            final_scores = torch.cat(all_scores, dim=1)
        else:
            final_scores = None
            
        # Truncate to max_len
        if final_input_ids.shape[1] > max_len:
            final_input_ids = final_input_ids[:, :max_len]
            
        if final_scores is not None:
            num_new_tokens = final_input_ids.shape[1] - input_ids.shape[1]
            if final_scores.shape[1] > num_new_tokens:
                final_scores = final_scores[:, :num_new_tokens, :]
        
        # print(f"before retrun: {final_input_ids[:, input_ids.shape[1]:]}")
        return {
            "sequences": final_input_ids,
            "scores": final_scores,
            "num_tokens": final_input_ids.shape[1] - input_ids.shape[1],
            "num_step": step_count
        }

    @torch.no_grad()
    def autoregressive_generate(
        self,
        input_ids: torch.LongTensor,
        max_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        eos_token_id: int = None,
    ):
        # print("in medusa's autoregressive generate")
        from sampling.utils import norm_logits, sample
        
        x = input_ids
        seq_len = x.shape[1]
        n = 0
        past_key_values = None
        scores = []
        
        while x.shape[1] < max_len:
            n += 1
            if past_key_values:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = torch.unsqueeze(last_ids, 0)
                outputs = self.base_model.model(last_ids, past_key_values=past_key_values, use_cache=True)
            else:
                outputs = self.base_model.model(x, use_cache=True)
            
            hidden_states = outputs[0]
            logits = self.base_model.lm_head(hidden_states)
            
            last_p = norm_logits(logits[::, -1, :], temperature, top_k, top_p)
            scores.append(logits[::, -1, :])
            past_key_values = outputs.past_key_values
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)
            
            if eos_token_id is not None and idx_next[0] == eos_token_id:
                break
                
        return {
            "sequences": x,
            "scores": torch.stack(scores, dim=1) if scores else None,
            "num_tokens": x.shape[-1] - seq_len,
            "num_step": n,
        }

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
        # print("from_pretrained")
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
            return MedusaModelOlmo.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
            
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type
        raise ValueError("Only support llama and olmo for now!!")

