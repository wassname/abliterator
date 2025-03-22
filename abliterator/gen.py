import torch
from torch import Tensor
from typing import List, Tuple, Dict, Callable
import collections
from tqdm.auto import tqdm
from jaxtyping import Int, Float
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict


@torch.no_grad()
def get_generations(
    instructions: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, "batch_size seq_len"]],
    layer_names: List[str] = [],
    max_new_tokens: int = 64,
    batch_size: int = 4,
    edit_output: Callable[
        [Float[Tensor, "batch_size seq_len dim"], str],
        Float[Tensor, "batch_size seq_len dim"],
    ] = None,
) -> Tuple[Dict[str, Float[Tensor, "batch tokens dim"]], List[str]]:
    """
    Generate a completion, optionall gathering activations, or editing the model using baukit
    """
    generations = []
    activations = collections.defaultdict(list)
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(
            instructions=instructions[i : i + batch_size]
        ).to(device)

        # record activations from just the next token
        # docs for TraceDict here: https://github.com/davidbau/baukit/blob/main/baukit/nethook.py
        with TraceDict(
            model, layers=layer_names, edit_output=edit_output,
        ) as ret:
            model(**inputs)

        for layer_name in layer_names:
            act = ret[layer_name].output[0].cpu()
            activations[layer_name].append(act)

        with TraceDict(
            model, layers=layer_names, edit_output=edit_output, retain_output=False,
        ) as ret2:
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens)
        t = inputs.input_ids.shape[1]
        generation = generation[:, t:]
        generations.extend(generation)

    pos = -1  # just the last token
    activations = {
        k: torch.concatenate([vv[:, pos] for vv in v], dim=0).cpu()
        for k, v in activations.items()
    }
    generations = tokenizer.batch_decode(generations, skip_special_tokens=True)

    return activations, generations
