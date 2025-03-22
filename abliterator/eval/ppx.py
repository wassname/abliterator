import torch
import numpy as np
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PreTrainedTokenizerBase

@torch.no_grad()
def compute_perplexity(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, stride=8, max_length=512, batch_size=2):
    """
    Efficient corpus perplexity calculation using strided windows.
    
    Args:
        model: A pretrained language model
        tokenizer: The tokenizer used to preprocess the data
        dataset: A dataset to calculate perplexity on. If None, the wikitext-2 test set is used.
        stride: The stride to use for perplexity calculation - Important, changing this will change your results
        max_length: The maximum length of each window, this will change your results
        batch_size: The batch size to use for perplexity calculation
        
    Comparison again other implementations:
    - https://huggingface.co/docs/transformers/perplexity - takes the mean of means giving it the wrong value
    - https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py - compelx and crops sentances so it's not comparable
    - https://github.com/ggerganov/llama.cpp/tree/master/examples/perplexity - good but in cpp
    - https://github.com/huggingface/transformers/issues/9648#issuecomment-812981524 - doesn't use special tokens
    """
    device = model.device
    
    # Tokenize corpus
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    seq_len = encodings.input_ids.size(1)
    
    # Initialize tracking variables
    nlls, counts = 0, 0
    
    # Configure loss function
    loss_fn = CrossEntropyLoss(reduction="none")
    
    # Process corpus in strided windows
    for i in tqdm(range(0, seq_len, stride * batch_size)):
        # Prepare batch windows
        input_ids_list, target_masks_list = [], []
        
        for j in range(batch_size):
            # Window start position
            start_idx = i + j * stride
            if start_idx >= seq_len:
                break
                
            # Extract window with context
            end_idx = min(start_idx + max_length, seq_len)
            ids = encodings.input_ids[0, start_idx:end_idx].clone()
            
            # Skip windows that are too small
            if len(ids) < 2:
                continue
                
            # Add BOS token for initial window
            if start_idx == 0:
                ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), ids])
            
            # Create evaluation mask (1 for tokens to evaluate, 0 otherwise)
            # For overlapping windows, only evaluate tokens beyond the overlap point
            eval_mask = torch.zeros_like(ids)
            eval_offset = 0 if start_idx == 0 else stride
            eval_mask[eval_offset:] = 1
            
            input_ids_list.append(ids)
            target_masks_list.append(eval_mask)
        
        if not input_ids_list:
            continue
            
        # Create padded batch tensors
        batch = tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Create padded target masks
        max_len = input_ids.size(1)
        padded_masks = []
        for mask in target_masks_list:
            padding = torch.zeros(max_len - len(mask), dtype=torch.long)
            padded_masks.append(torch.cat([mask, padding]))
        target_masks = torch.stack(padded_masks).to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Compute loss on shifted sequences
        shift_logits = outputs.logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_masks = target_masks[:, 1:].contiguous() * attention_mask[:, 1:].contiguous()
        
        # Calculate NLL only for targeted tokens
        loss = loss_fn(shift_logits.transpose(1, 2), shift_labels)
        masked_loss = (loss * shift_masks).sum()
        token_count = shift_masks.sum()
        
        # Accumulate results
        nlls += masked_loss.item()
        counts += token_count.item()
    
    # Return corpus-level perplexity
    s = np.exp(nlls / counts) if counts > 0 else float('inf')
    return float(s)

