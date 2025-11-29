import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, eos_token_id : int = None):
    seq_len = x.shape[1]
    T = seq_len + N
    n=0
    past_key_values = None
    scores = []
    while x.shape[1] < T:
        n += 1

        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        scores.append(outputs.logits[::, -1, :])
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        
        if eos_token_id is not None and idx_next[0] == eos_token_id:
            # print("end eos")
            break
        
    return {
        "sequences": x,
        "scores": torch.stack(scores, dim=1),
        "num_tokens": x.shape[-1] - seq_len,
        "num_step": n,
        "generation": x[:, seq_len:]
    }

