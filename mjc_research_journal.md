# 2025-03-21 11:26:52

Using paired prompts works better (mean over tokens, then substract, then mean). This way I'm subtraction one prompt from it's pair eather than a cluster of prompts from another cluster.


unsloth/llama-3.2-3b-instruct
| model       |   perplexity |   ppx_ratio |
|:------------|-------------:|------------:|
| base        |      8.63868 |    0.145753 |
| edit_output |      8.9129  |    0.154275 |

get_first_non_zero_token

    unsloth/llama-3.2-3b-instruct
    | model       |   perplexity |   ppx_ratio |
    |:------------|-------------:|------------:|
    | base        |      8.63868 |    0.145753 |
    | edit_output |      8.50475 |    0.152796 |

mean agg
    unsloth/llama-3.2-3b-instruct
    | model       |   perplexity |   ppx_ratio |
    |:------------|-------------:|------------:|
    | base        |      8.63868 |    0.145753 |
    | edit_output |      8.9129  |    0.155002 |

last (winner!)
    unsloth/llama-3.2-3b-instruct
    | model       |   perplexity |   ppx_ratio |
    |:------------|-------------:|------------:|
    | base        |      8.63868 |    0.145753 |
    | edit_output |      8.77472 |    0.161948 |
