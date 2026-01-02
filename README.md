# wassname's Abliterator (with baukit not transformerlens)

What is this? It's a concept removal method, here I show how to use baukit, which is a nice library which uses pytorch hooks to modify layers at runtime.

```sh
uv sync
```

then open nbs/01_abliterate.ipynb in Jupyter Notebook.


TODO:
- [ ] tidy up
- [ ] use https://github.com/wassname/activation_store to cache large numbers of examples to disc


disabled-docs/image.png
## See also


I use
- performance: wikitext perplexity
- compliance: https://huggingface.co/datasets/wassname/genies_preferences

- FailSpy's abliterator: https://github.com/FailSpy/abliterator
  - uses datasets `Undi95/orthogonal-activation-steering-TOXIC` vs `tatsu-lab/alpaca`
- a more advanced method, instead of removing all diferences, it removes the ones that are predictive, https://github.com/EleutherAI/concept-erasure https://github.com/EleutherAI/conceptual-constraints/blob/main/notebooks/concept_erasure.ipynb
  - which uses the [HANS](https://arxiv.org/abs/1902.01007) dataset:
- https://github.com/Sumandora/remove-refusals-with-transformers/
  - uses `advbench/harmful_behaviors` vs `alpaca_cleaned`
- https://huggingface.co/blog/mlabonne/abliteration
