import torch
from typing import Optional

from sampling.utils import norm_logits, sample
# from transformers.models.bloom.modeling_bloom import BloomForCausalLM # Bloom support removed

def _debug_show_kvcache(past_key_values):
    if past_key_values is None:
        return
    # Assuming DynamicCache
    print(f"kv cache length: {past_key_values.get_seq_length()}")
    
class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._logit_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True, use_base_model_only = False) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            if use_base_model_only and hasattr(self._model, "base_model_forward"):
                outputs = self._model.base_model_forward(input_ids, use_cache=True)
            else:
                outputs = self._model(input_ids, use_cache=True)
            self._logit_history = outputs.logits.clone()
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            # DynamicCache support
            cached_len = self._past_key_values.get_seq_length()
            
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            if use_base_model_only and hasattr(self._model, "base_model_forward"):
                outputs = self._model.base_model_forward(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            else:
                outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            self._logit_history = torch.cat([self._logit_history, outputs.logits], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        medusa_logits = None
        if hasattr(outputs, "medusa_logits"):
            medusa_logits = outputs.medusa_logits
            
        return last_q, medusa_logits


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False,
                                    use_base_model_only = True) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        # Try to generate gamma tokens in one go if medusa_logits are available
        q, medusa_logits = self._forward_with_kvcache(x, use_debug, use_base_model_only=use_base_model_only)
        next_tok = sample(q)
        x = torch.cat((x, next_tok), dim=1)
        
        if medusa_logits is not None and not use_base_model_only:
       
             for i in range(min(len(medusa_logits), gamma - 1)):
                 # medusa_logits[i] shape: (batch, seq_len, vocab)
                 # We want the prediction for the last position
                 m_logits = medusa_logits[i][:, -1, :]
                
                 # We need the raw logits before normalization for scores
                 self._logit_history = torch.cat([self._logit_history, m_logits.unsqueeze(1)], dim=1)
                 m_logits = norm_logits(m_logits, self._temperature, self._top_k, self._top_p)
                 
                 # Update prob_history
                 # m_logits is (1, vocab), we need to unsqueeze to match (1, 1, vocab) for cat
                 self._prob_history = torch.cat([self._prob_history, m_logits.unsqueeze(1)], dim=1)
                 
                 m_tok = sample(m_logits)
                 x = torch.cat((x, m_tok), dim=1)
                 
        else:
            # print("use base logit")

            # Fallback to sequential generation
            for _ in range(gamma - 1):
                q, _ = self._forward_with_kvcache(x, use_debug)
                next_tok = sample(q)
                x = torch.cat((x, next_tok), dim=1)
        
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int, use_base_model_only = False) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma, use_base_model_only=use_base_model_only)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        assert self._past_key_values

        # Support for DynamicCache
        new_key_cache = []
        new_value_cache = []
        
        # Iterate over layers
        for k, v in zip(self._past_key_values.key_cache, self._past_key_values.value_cache):
            k = k[..., :end_pos, :]
            v = v[..., :end_pos, :]
        
            new_key_cache.append(k)
            new_value_cache.append(v)
        
        self._past_key_values.key_cache = new_key_cache
        self._past_key_values.value_cache = new_value_cache
        
        self._past_key_values._seen_tokens = end_pos
        
        self._prob_history = self._prob_history[:, :end_pos, :]
        self._logit_history = self._logit_history[:, :end_pos, :]

    def sync_from(self, other):
        """
        Syncs the KV cache and history from another KVCacheModel.
        Used to update the approx model's cache with the target model's cache after verification.
        """
        
        import copy
        # Shallow copy the object to get a new DynamicCache instance
        self._past_key_values = copy.copy(other._past_key_values) 
        
        # Shallow copy the lists. The tensors inside are shared.
        # This is safe because cache updates usually create new tensors (torch.cat) 
        # or slice existing ones (creating views), rather than modifying in-place.
        if hasattr(other._past_key_values, 'key_cache'):
            self._past_key_values.key_cache = list(other._past_key_values.key_cache)
        if hasattr(other._past_key_values, 'value_cache'):
            self._past_key_values.value_cache = list(other._past_key_values.value_cache)

        if hasattr(other._past_key_values, '_seen_tokens'):
            self._past_key_values._seen_tokens = other._past_key_values._seen_tokens
    
        self._prob_history = other._prob_history.clone()
        self._logit_history = other._logit_history.clone()
