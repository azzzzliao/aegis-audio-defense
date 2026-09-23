# Representations

Where attacks and benign inputs sit inside the frozen model, and what the defense moves.
Every figure here is computed from hidden states extracted with `aegis/extract_hidden.py`.

## Attacks are not one cluster

![t-SNE of hidden states by benchmark](../../assets/figures/tsne_benchmarks_allmodels.png)

Hidden states coloured by benchmark. The three attack families occupy different regions:
they are different input distributions, not variations of one attack. This is the reason
the paper holds a whole family out rather than splitting one benchmark, and the reason a
defense fitted to one family need not transfer.

![t-SNE of projected audio embeddings](../../assets/figures/tsne_projected_audio.png)

The same picture at the audio projector output, before the decoder: part of the
separation is already present in what the audio tower hands to the language model.

## What the defense moves

![PCA before and after](../../assets/figures/pca_before_after.png)

Attack and benign points in the top principal components of a late-layer hidden state,
before and after the intervention. Attack points move; benign points stay where they
were, which is the gate doing its job — with a low score the residual update is
multiplied by roughly zero.

| | | |
|---|---|---|
| ![Phi-4](../../assets/figures/pca_before_after_phi4.png) | ![Ultravox](../../assets/figures/pca_before_after_ultravox.png) | ![VITA-1.5](../../assets/figures/pca_before_after_vita.png) |

## Layer-wise separability

![Separability by benchmark](../../assets/figures/hidden_separability_gemma_3bench.png)

Per layer, an MLP probe on the last-input-token hidden state predicts the Llama-Guard
label of the response, scored by 5-fold CV AUROC, one line per benchmark. Separability is
low in the shallow layers, rises through the middle, and does not peak at the last layer —
which is what puts the gate in the middle and the intervention late.

| | |
|---|---|
| ![Qwen2-Audio per benchmark](../../assets/figures/hidden_separability_qwen_perbench.png) | ![MLP probe](../../assets/figures/hidden_separability_mlp.png) |
