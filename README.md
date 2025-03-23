# wassname's Abliterator (with baukit not transformerlens)


```sh
uv sync
```

then open nbs/01_abliterate.ipynb in Jupyter Notebook.


TODO:
- [ ] tidy up
- [ ] use https://github.com/wassname/activation_store to cache large numbers of examples to disc


![alt text](docs/image.png)


> TODO update. mention the main features and advantages: caching activations to disk to allow us to use large datasets, using logistic regression to find a **predictive** direction, using pair preference data to allow us to compare invidiual prompts (a more surgical approach), using baukit instead of transformerlens to make it (in my opinion) lighter

## See also


I use
- performance: wikitext perplexity
- compliance: https://huggingface.co/datasets/wassname/genies_preferences

See also
- activation steering
  - https://github.com/annahdo/implementing_activation_steering
  - https://github.com/vgel/repeng
  - https://sidn.baulab.info/steering/
- albiteration
  - FailSpy's abliterator: https://github.com/FailSpy/abliterator
    - uses datasets `Undi95/orthogonal-activation-steering-TOXIC` vs `tatsu-lab/alpaca`
  - a more advanced method, instead of removing all diferences, it removes the ones that are predictive, https://github.com/EleutherAI/concept-erasure https://github.com/EleutherAI/conceptual-constraints/blob/main/notebooks/concept_erasure.ipynb
    - which uses the [HANS](https://arxiv.org/abs/1902.01007) dataset:
  - https://github.com/Sumandora/remove-refusals-with-transformers/
    - uses `advbench/harmful_behaviors` vs `alpaca_cleaned`
  - https://huggingface.co/blog/mlabonne/abliteration

Status 2025-03-23 11:53:07: nothing works well or reliably, and I think we need better quick dev measurements too

TODO:
- try getting the control vector from an A/B type token at the end https://arxiv.org/abs/2312.06681
- I think that overall it's must be a non-linear problem and gradient methods like Fine Tuning a LoRA adapter to give particular representations, that seems the most promising (e.g. https://github.com/wassname/repr-preference-optimization). In general we should use powerful tools like gradient descent.
