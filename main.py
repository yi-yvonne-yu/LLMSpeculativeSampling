
import torch
import argparse
# import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder
from medusa_model import MedusaModel, MedusaConfig
from transformers import AutoConfig

AutoConfig.register("medusa", MedusaConfig)
AutoModelForCausalLM.register(MedusaConfig, MedusaModel)

import time

# my local models
MODELZOO = {
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama2_1b" : "meta-llama/Llama-3.2-1B",
    "llama2_7b" : "meta-llama/Llama-2-7b-hf",
    "llama2_13b" : "meta-llama/Llama-2-13b-hf",
    "llama2_70b" : "meta-llama/Llama-2-70b-hf",
    "superbpe_3h_stage2" : "yvonne90190/superbpe_1b_3h_stage2",
    "superbpe_5h_stage2" : "yvonne90190/superbpe_1b_5h_stage2",
    "medusa_1b_7h_stage2" : "yvonne90190/medusa_1b_7h_stage2",
    "medusa_1b_5h_stage2" : "yvonne90190/medusa_1b_5h_stage2",
    "medusa_1b_3h_stage2" : "yvonne90190/medusa_1b_3h_stage2",
    "olmo2" : "allenai/OLMo-2-0425-1B-Instruct"
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Where can I go in COlumbus?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["medusa_1b_7h_stage2"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["medusa_1b_7h_stage2"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=2048, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    start_time = time.perf_counter()
    if use_profiler:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
            record_shapes=False,
            profile_memory=False,
            # with_stack=True
        ) as prof:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)
                prof.step()
    else:
        for _ in range(TEST_TIME): 
            output = fn(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    if isinstance(output, dict):
        num_tokens = output["num_tokens"]
    else:
        num_tokens = len(output[0])

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {num_tokens / elapsed * TEST_TIME}, {elapsed / TEST_TIME} sec generates {num_tokens} tokens")

def generate(input_text, approx_model_name, target_model_name, max_tokens=2048, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    print("finish loading models")
    
    if hasattr(small_model.config, "medusa_num_heads"):
        gamma = small_model.config.medusa_num_heads + 1
        # print(f"gamma is set to {gamma} from medusa_num_heads")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    # torch.manual_seed(123)
    # start_time = time.time()
    # output, num_tokens = autoregressive_sampling(input_ids, large_model, max_tokens, top_k = top_k, top_p=top_p)
    # total_time = time.time() - start_time
    # print(f"num_tokens = {num_tokens}, time = {total_time}, token/sec = {num_tokens/total_time}")
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_large", use_profiling,
    #               input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    start_time = time.time()
    if hasattr(small_model, "autoregressive_generate"):
        print("Using model.autoregressive_generate")
        output = small_model.autoregressive_generate(input_ids, max_len=max_tokens, top_k=top_k, top_p=top_p, eos_token_id=tokenizer.eos_token_id)
    else:
        output = autoregressive_sampling(input_ids, small_model, max_tokens, top_k = top_k, top_p=top_p, eos_token_id=tokenizer.eos_token_id)
    num_tokens = output["num_tokens"]
    sequences = output["sequences"]
    total_time = time.time() - start_time
    print(f"time = {total_time}")
    print(f"num_tokens = {output['num_tokens']}")
    print(f"scores.shape = {output['scores'].shape}")
    print(f"num_step = {output['num_step']}")
    print(f"token/sec = {num_tokens/total_time}")

    generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, max_tokens, top_k = top_k, top_p=top_p)
    
    # torch.manual_seed(123)
    # output, num_tokens = speculative_sampling_v2(input_ids, small_model, large_model, max_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    # total_time = time.time() - start_time
    # print(f"num_tokens = {num_tokens}, time = {total_time}, token/sec = {num_tokens/total_time}")
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"deepmind's speculative_sampling: {generated_text}")   

    if hasattr(small_model, "medusa_generate"):
        print("Using optimized medusa_generate")
        torch.manual_seed(123)
        start_time = time.time()
        output = small_model.medusa_generate(input_ids, max_len=max_tokens, gamma=gamma, top_k=top_k, top_p=top_p, eos_token_id=tokenizer.eos_token_id)
        sequences = output["sequences"]
        total_time = time.time() - start_time
        num_tokens = sequences.shape[1] - input_ids.shape[1]
        print(f"time = {total_time}")
        print(f"num_tokens = {output['num_tokens']}")
        print(f"scores.shape = {output['scores'].shape}")
        print(f"num_step = {output['num_step']}")
        print(f"token/sec = {num_tokens/total_time}")
        generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        color_print(f"medusa optimized sampling: {generated_text}")
    else:
        torch.manual_seed(123)
        start_time = time.time()
        output = speculative_sampling(input_ids, small_model, large_model, max_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose, eos_token_id=tokenizer.eos_token_id)
        num_tokens = output["num_tokens"]
        sequences = output["sequences"]
        total_time = time.time() - start_time
        print(f"time = {total_time}")
        print(f"num_tokens = {output['num_tokens']}")
        print(f"scores.shape = {output['scores'].shape}")
        print(f"num_step = {output['num_step']}")
        print(f"token/sec = {num_tokens/total_time}")
        generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = max_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.input, args.approx_model_name, args.target_model_name, max_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
