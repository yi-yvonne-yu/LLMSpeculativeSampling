import torch
from typing import Optional

from sampling.utils import norm_logits, sample
# from transformers.models.bloom.modeling_bloom import BloomForCausalLM # Bloom support removed

def _debug_show_kvcache(past_key_values):
    if past_key_values is None:
        return
    # Assuming DynamicCache
    if hasattr(past_key_values, 'get_seq_length'):
        print(f"kv cache length: {past_key_values.get_seq_length()}")
    elif isinstance(past_key_values, list) or isinstance(past_key_values, tuple):
        # Fallback for some legacy or other types if they slip through, though we mainly support DynamicCache now
        print(f"kv cache (tuple/list) length: {len(past_key_values)}")

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
            if hasattr(self._past_key_values, 'get_seq_length'):
                cached_len = self._past_key_values.get_seq_length()
            else:
                 # Fallback/Error if not DynamicCache (since we removed tuple support, this is just a safety check or crash)
                 raise ValueError("past_key_values does not appear to be a DynamicCache (missing get_seq_length)")

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
            
            #  print("use medusa logit")
             # Medusa generation: use medusa heads to predict subsequent tokens
             # medusa_logits is a list of tensors, each predicting x_{t+2+k}
             # We need to generate gamma-1 more tokens (since we already generated 1)
             # But wait, medusa_logits[k] predicts the token at offset k+1 from the current position?
             # Let's verify: medusa_logits[0] predicts x_{t+2}
             # We just generated x_{t+1} (next_tok)
             # So we can use medusa_logits to generate x_{t+2}, x_{t+3}, ...
             
             # Ensure we don't generate more than gamma tokens total
             # We already generated 1.
             
             for i in range(min(len(medusa_logits), gamma - 1)):
                 # medusa_logits[i] shape: (batch, seq_len, vocab)
                 # We want the prediction for the last position
                 m_logits = medusa_logits[i][:, -1, :]
                 m_logits = norm_logits(m_logits, self._temperature, self._top_k, self._top_p)
                 
                 # Update prob_history
                 # m_logits is (1, vocab), we need to unsqueeze to match (1, 1, vocab) for cat
                 self._prob_history = torch.cat([self._prob_history, m_logits.unsqueeze(1)], dim=1)
                 
                 # Update logit_history
                 # We need the raw logits before normalization for scores
                 raw_m_logits = medusa_logits[i][:, -1, :]
                 self._logit_history = torch.cat([self._logit_history, raw_m_logits.unsqueeze(1)], dim=1)
                 
                 m_tok = sample(m_logits)
                 x = torch.cat((x, m_tok), dim=1)
                 
        else:
            # print("use base logit")

            # Fallback to sequential generation
            for _ in range(gamma - 1):
                q, _ = self._forward_with_kvcache(x, use_debug, use_base_model_only=use_base_model_only)
                next_tok = sample(q)
                x = torch.cat((x, next_tok), dim=1)
        
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int, use_base_model_only = False) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma, use_base_model_only=use_base_model_only)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        if self._past_key_values is None:
            return

        # Support for DynamicCache
        if hasattr(self._past_key_values, 'key_cache') and hasattr(self._past_key_values, 'value_cache'):
            new_key_cache = []
            new_value_cache = []
            
            # Iterate over layers
            for k, v in zip(self._past_key_values.key_cache, self._past_key_values.value_cache):
                # k, v shape: [batch, num_heads, seq_len, head_dim]
                # We slice the sequence length dimension (usually dim 2)
                # But wait, let's check shape. Usually it is (batch, num_heads, seq_len, head_dim)
                # end_pos is the absolute position to keep up to.
                
                # Safety check for dimensions
                if k.dim() >= 3:
                    k = k[..., :end_pos, :]
                    v = v[..., :end_pos, :]
                
                new_key_cache.append(k)
                new_value_cache.append(v)
            
            self._past_key_values.key_cache = new_key_cache
            self._past_key_values.value_cache = new_value_cache
            
            if hasattr(self._past_key_values, '_seen_tokens'):
                self._past_key_values._seen_tokens = end_pos
        else:
             raise ValueError("past_key_values does not appear to be a DynamicCache (missing key_cache/value_cache)")
        
        self._prob_history = self._prob_history[:, :end_pos, :]
        if self._logit_history is not None:
            self._logit_history = self._logit_history[:, :end_pos, :]

    def sync_from(self, other):
        """
        Syncs the KV cache and history from another KVCacheModel.
        Used to update the approx model's cache with the target model's cache after verification.
        """
        if other._past_key_values is None:
            self._past_key_values = None
        else:
            # Deep copy the DynamicCache
            if hasattr(other._past_key_values, 'key_cache') and hasattr(other._past_key_values, 'value_cache'):
                # We can't just copy the object because we might modify it later
                # But wait, we can just assign it if we assume the other model won't use it anymore in a conflicting way?
                # Actually, in speculative sampling, target_model generates 1 token, then we rollback.
                # Then we sync approx to target.
                # Then approx generates.
                # Target generates.
                # They don't run in parallel.
                # However, to be safe and avoid side effects if they share the same tensor objects and modify them in-place (unlikely for cache which usually appends),
                # let's try to do a shallow copy of the list but keep tensors shared (since they are immutable-ish in this context, we just append new ones).
                # Actually, DynamicCache usually appends to the list.
                
                # Let's try to create a new DynamicCache-like structure or just copy the lists.
                # Since we don't have the DynamicCache class constructor here easily, let's assume we can just copy the internal lists.
                
                # But wait, self._past_key_values might be None or a different object.
                # Let's just assign the object for now, but we need to be careful.
                # If we assign `self._past_key_values = other._past_key_values`, they point to the same object.
                # If `self` modifies it (appends), `other` sees it.
                # In speculative sampling, `approx` generates (appends). `target` verifies (computes from scratch or appends).
                # If `approx` appends to the SAME cache object that `target` has, then `target`'s cache is also modified.
                # When `target` runs verification, it might be confused if its cache grew unexpectedly?
                # Actually, `target` verification usually starts from the prefix.
                
                # Better to deep copy the lists of the cache.
                import copy
                self._past_key_values = copy.copy(other._past_key_values) # Shallow copy of the object
                self._past_key_values.key_cache = [k.clone() for k in other._past_key_values.key_cache]
                self._past_key_values.value_cache = [v.clone() for v in other._past_key_values.value_cache]
                if hasattr(other._past_key_values, '_seen_tokens'):
                     self._past_key_values._seen_tokens = other._past_key_values._seen_tokens
            else:
                 raise ValueError("past_key_values does not appear to be a DynamicCache")

        if other._prob_history is not None:
            self._prob_history = other._prob_history.clone()
        else:
            self._prob_history = None
            
        if other._logit_history is not None:
            self._logit_history = other._logit_history.clone()
        else:
            self._logit_history = None
