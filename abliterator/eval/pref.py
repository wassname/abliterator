"""
This is a simple way to evaluate if a model prefers the accepted or rejected completions of a prompt.

We look at the perplexity of the chosen and rejected completions of a prompt.

Example dataset: https://huggingface.co/datasets/wassname/genies_preferences/viewer/illegal_dont_help?views[]=illegal_dont_help_train&views[]=illegal_dont_help_test

@url: https://gist.github.com/wassname/04f0c50a68054f0323f62b0da418daec
"""
import pandas as pd
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import torch
import numpy as np
import copy
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset

# how to eval, I couldlook at perplexity on chosen vs rejected in the context of prompt

def get_output_ppx(output, input):
    loss_fn = CrossEntropyLoss(reduction="none")
    shift_logits = output.logits[:, :-1].contiguous()
    shift_labels = input.input_ids[:, 1:].contiguous()
    loss = loss_fn(shift_logits.transpose(1, 2), shift_labels)

    # crop the attention mask to just the provided input
    attention_mask = input.attention_mask[:, :input.input_ids.size(1)].contiguous()
    # input.attention_mask
    shift_masks = attention_mask[:, 1:].contiguous()
    nll = (loss * shift_masks)
    count = shift_masks.sum().item()
    return {
        'ppx': np.exp(nll.sum().item() / count),
        # 'nll': nll.sum().item(),
        'nll_mean': nll.sum().item() / count,
        # 'count': count,
    }


@torch.no_grad()
def eval_pref_ds_ppx(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, ds_pref: Dataset, batch_size: int=2, max_new_tokens: int=128):
    """
    Evaluate on a preference dataset. 
    
    The relative perplexity of the chosen and rejected completions of a prompt.
    """
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    results = []
    for batch in tqdm(ds_pref.batch(batch_size), unit="batch"):
        # first we cache the prompt
        kv_cache = DynamicCache()
        inputs1 = tokenizer(batch['prompt'], return_tensors="pt", padding=True, truncation=True, max_length=max_new_tokens//2, return_token_type_ids=False, return_attention_mask=True).to(model_device)
        model.forward(**inputs1, past_key_values=kv_cache)

        # then we evaluate the perplexity of the accepted and rejected completion
        res = {}
        for p in ['rejected', 'chosen']:
            input = tokenizer(batch[p], return_tensors="pt", padding=True, truncation=True, max_length=max_new_tokens//2, return_token_type_ids=False, return_attention_mask=True).to(model_device)

            # we need to update the attention mask to match the kv_cache
            input['attention_mask'] = torch.cat([inputs1['attention_mask'], input['attention_mask']], dim=1)

            kv_cache2 = copy.deepcopy(kv_cache)
            output = model.forward(**input, past_key_values=kv_cache2)
            ppx = get_output_ppx(output, input)
            for k in ppx:
                res[f"{p}_{k}"] = ppx[k]
        results.append(res)

    df = pd.DataFrame(results)
    df['ppx_ratio'] = (df.chosen_ppx/df.rejected_ppx)
    # df['ppx_ratio'] = (df.chosen_nll-df.rejected_nll)
    return df



