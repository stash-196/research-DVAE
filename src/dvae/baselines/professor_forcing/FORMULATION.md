# Professor Forcing on Lorenz63 — what follows the paper

Sources used while writing this baseline:

- Lamb, Goyal, Zhang, Zhang, Courville, and Bengio, *Professor Forcing: A New Algorithm for Training Recurrent Networks*, NeurIPS 2016. arXiv:1610.09038. Equations below are those in sections 2.2 and 4.1 of that paper.
- The authors' Theano code: [anirudh9119/LM_GANS](https://github.com/anirudh9119/LM_GANS), branch `updated_generator` (the README on `master` says the repository is under construction and to use that branch). Files consulted: `lm_char_worker.py`, `Descriminator.py`, `working_desc_hidden.py`, `lm_discriminator.py`, `config_pbt.json`.

This file lists what is taken from the paper and every place the Lorenz implementation differs. Nothing here is filled in from memory of a detail the paper does not state.

## What follows the paper

**Generator.** One GRU layer (section 4.1). At step \(t\) the network emits a prediction from the hidden state that has seen the preceding inputs, then updates that state with either the observed target (teacher forcing) or its own prediction (free running). The same parameters are used in both modes. The cell is the standard reset / update / candidate GRU, and the behavior sequence uses the candidate's pre-tanh activation, which is the quantity section 4.1 calls "the pre-tanh activation of the GRU states." A unit test checks that the hidden state matches `torch.nn.GRUCell` when the weights are copied.

**Behavior sequences fed to the discriminator** (section 4.1, and the character-LM note in section 4.2):

| `behavior` | Contents |
| --- | --- |
| `pre_tanh` | Pre-tanh candidate activation. This is the section 4.1 default. |
| `pre_tanh_and_output` | That activation concatenated with the next-step emission. Section 4.1 says the softmax outputs are optional. |
| `hidden` | Post-update hidden state \(h_t\). |
| `hidden_and_output` | \(h_t\) concatenated with the next-step emission. |
| `hidden_and_input` | \(h_t\) concatenated with the vector that was fed into the GRU at that step. |

Section 4.2 says the Penn Treebank run "feed[s] both the hidden states and char level embeddings into the discriminator." Lorenz has no embedding table. `hidden_and_input` uses the continuous vector that entered the GRU in place of that embedding. `hidden_and_output` uses the emission instead. Neither is claimed to be the character-LM input.

The next-step emission paired with the state at time \(t\) is the readout of the hidden state *after* the GRU consumes the input at time \(t\). That is \(P(y_{t+1} \mid h_t)\) in the paper's indexing. The training NLL compares the readout *before* that update with the target at time \(t\).

**Discriminator** (section 4.1). Two GRUs read the behavior sequence, one forward and one backward, with separate parameters. At each time their hidden states are concatenated and passed through an MLP that is shared across time: three affine layers, each followed by a ReLU, then an affine output. The per-step logits are averaged, clipped to \([-10, 10]\), and passed through a sigmoid. The clip is the character-LM setting in section 4.2 ("Output activation of the last layer is clipped between -10 and 10"), not a statement in section 2. Set `logit_clip = 0` to turn it off.

The paper writes a single \(D(b)\) but also says the MLP is applied at every time. Averaging the pre-sigmoid logits is the reduction used so that \(D(b)\) is one probability per sequence, as in equations (1)–(4). The released `Descriminator.py` instead averages the GRU states and then applies a shallower MLP. That reduction is not what this code does.

**Losses** (section 2.2). \(D(b)\) is the estimated probability that the behavior came from teacher forcing.

- \(C_d = -\log D(b_{\mathrm{TF}}) - \log(1 - D(b_{\mathrm{free}}))\)
- \(\mathrm{NLL}\) on the teacher-forced predictions only
- \(C_f = -\log D(b_{\mathrm{free}})\)
- \(C_t = -\log(1 - D(b_{\mathrm{TF}}))\), optional

`generator_objective = nll_cf` trains the generator on \(\mathrm{NLL} + C_f\). `nll_cf_ct` trains it on \(\mathrm{NLL} + C_f + C_t\). The discriminator is trained on \(C_d\). Minibatches contain the teacher-forced behavior and the free-running behavior of the same sequences (equal counts). Classification accuracy is the fraction of those \(2N\) sequences for which \(D > 0.5\) agrees with the teacher-forcing label.

**Accuracy gate** (section 4.1, verbatim in effect):

- Adversarial gradients are sent into the generator only when accuracy is **greater than 75%**.
- The discriminator is **not** updated when accuracy is **greater than 99%**.

These are separate tests. Above 99% the generator still receives \(C_f\) (and \(C_t\) if selected). At or below 75% the generator receives only the NLL, and the discriminator still updates. At exactly 75% the adversarial term is off. At exactly 99% the discriminator still updates. The gate uses the current minibatch, measured with stop-gradient before the parameter step.

**Optimizer.** Both networks use Adam (Kingma and Ba, 2014), which is what section 4.1 states. The learning rate defaults to \(10^{-4}\), the Penn Treebank setting in section 4.2. Betas are PyTorch's defaults (0.9, 0.999). The same rate is used for the generator and the discriminator unless `disc_lr` is changed.

## Continuous Lorenz: free-running feedback and the NLL

The paper's experiments are discrete or binned (character LM, binarized MNIST, 8000-bin waveforms) or a handwriting mixture-density model that samples. It does not define an emission for a continuous trajectory such as Lorenz63, and it does not say whether a Gaussian decoder should feed its mean or a sample back into the recurrence.

**Default feedback is the deterministic conditional mean** (`free_run_feedback = mean`). The scalar readout is the mean of a fixed-variance isotropic Gaussian, and that mean is what the next free-running step consumes. No noise is added. This is the closed loop that `mse_auto` measures.

`free_run_feedback = gaussian` is optional. The free-running input is then \(\hat y_t + \sigma \varepsilon_t\) with \(\varepsilon_t \sim \mathcal N(0, I)\) and \(\sigma =\) `observation_std` (default 1). The reparameterization is differentiable. The behavior feature still carries the mean, which is the parameter of the predictive distribution, analogous to feeding softmax probabilities rather than a sampled token. Evaluation rollouts ignore this switch and always feed the mean, so `mse_auto` stays a deterministic closed loop.

**NLL is the repo's masked squared error, not a normalized Gaussian log-density.** In `LearningAlgorithm`, Lorenz / SHO reconstruction is `loss_MSE` (the sum of squared residuals) divided by `seq_len * batch_size`. Masked coordinates are multiplied by zero before the sum; the denominator is still \(T \times B\), not the count of observed entries and not \(D\). This baseline uses that same quantity as the NLL term, computed on the teacher-forced emissions only.

For a fixed-variance isotropic Gaussian the negative log-density of one coordinate is \(\tfrac12\log(2\pi\sigma^2) + (y-\mu)^2 / (2\sigma^2)\). Dropping the constant and setting \(\sigma^2 = 1/2\) makes the sum of squares equal that sum of log-densities. The repo then divides by \(T \times B\) rather than by the number of summands. This code does not add the \(\tfrac12\log(2\pi\sigma^2)\) term and does not divide by \(2\sigma^2\). The training term is the repo loss itself, so its scale matches OTF training. It is proportional to a fixed-variance Gaussian NLL and is not equal to it.

## Deviations

1. **Closed loop versus `BaseRNN`.** `BaseModel.mix_inputs_with_outputs` always feeds the observation at \(t = 0\), and for later autonomous steps it feeds the *previous* emission. This generator feeds the emission just produced, including at \(t = 0\) when the whole sequence is free-running. Teacher-forced steps still consume the observation, so a teacher-forced pass matches the repo's open loop: \(\hat y_t\) depends on inputs \(0..t-1\), not on the input at \(t\). `half_half` therefore agrees with a pure teacher-forced run on the first half and diverges after the flip. `mse_auto` is masked MSE on that post-flip segment, which is the same statistic the eval helpers compute, under this closed loop rather than `BaseRNN`'s one-step-lagged feedback.

2. **Missing coordinates.** The paper assumes observed sequences. Where a teacher-forced target is non-finite, the input uses the repo's fill (`BaseModel.impute_inputs_nans_with_output`): zeros at \(t = 0\), otherwise the previous emission. Those coordinates are excluded from the squared error by the same zero-mask the trainer uses. Free-running steps do not read the observation at all.

3. **Indicate / partial observations.** `only_x`, `only_xy`, `only_xz`, and `all_xyz` are the dataset's column slices; the GRU input width is that slice. For `only_x_indicate` (and any `*_indicate` process) the layout is interleaved signal and indicator. Indicator channels are always copied from the data, which is the "100% teacher forcing on the mask channel" rule in `LearningAlgorithm._prepare_masked_loss`. The squared error is computed only on the even (signal) channels. That rule is from this repo's indicate baseline, not from the paper. `mse_*` for indicate also treats an indicator value below 0.5 as the missingness mask passed to `calc_masked_mse`, because the stored series has already been zero-filled and the eval mask is the original missingness bit.

4. **Discriminator width and depth relative to the released code.** Section 4.1 specifies a bidirectional GRU plus a 3-layer ReLU MLP. Penn Treebank uses 2048 units in every discriminator layer and a 1024-unit generator. Those widths are for that language model. Lorenz runs default to 64 generator units and 64 discriminator units. `Descriminator.py` mean-pools then uses two dense layers; `working_desc_hidden.py` uses a conv net and clips to \([-20, 20]\). This code follows section 4.1 and the \([-10, 10]\) clip from section 4.2.

5. **Time reduction.** Mean of per-step logits, then one sigmoid. The paper does not write the reduction. See above.

6. **No variable-length mask.** The language-model code carries an `x_mask`. Lorenz windows used here are rectangular, so the discriminator sees the full window.

7. **Accuracy gate versus `LM_GANS`.** `config_pbt.json` sets `limit_desc_start = 0.70` and `limit_desc_end = 0.90`. In `lm_char_worker.py` the gate that was actually consulted is a 0.9 EMA of accuracy compared with `limit_desc_end`: above that threshold the step updates the generator only; otherwise it updates both. `limit_desc_start` is an argument of that worker and is not read in the branch that was inspected. This baseline uses the paper's 75% and 99% tests on the current minibatch, not that EMA and not 0.70 / 0.90.

8. **Adam hyperparameters versus `LM_GANS`.** `config_pbt.json` sets `beta1 = 0.9` and `beta2 = 0.5`. The worker also multiplies the discriminator learning rate by 0.1 in one of the compiled updates. The paper cites Adam and does not state betas or a discriminator learning-rate multiplier. Defaults here are betas (0.9, 0.999), and `disc_lr = lr`.

9. **Gradient clip.** The paper does not mention clipping. `lm_char_worker.py` clips both generator and discriminator gradients with Lasagne's `total_norm_constraint(..., 5.0)`. `gradient_clip = 5` is that value. `gradient_clip = 0` disables it. There is no weight decay. The OTF trainer's Adam uses `weight_decay = 1e-5`; that coefficient is not in the paper and is not used here.

10. **Early stopping.** Patience counts evaluation checks on the validation teacher-forced repo loss. The paper monitors NLL / bits-per-character. `LM_GANS` early-stops on validation cost. The check interval is `eval_every`.

11. **Scheduled sampling is not implemented.** The paper discusses it as a different algorithm and does not mix it into Professor Forcing.

12. **Metrics.** `mse_tf`, `mse_auto`, `spectrum_error_*`, and `kld_*` are computed with the repo helpers listed in the package README, on a `half_half` split from `create_autonomous_mode_selector_1d`. They are summaries for comparison with OTF. They are not part of the training objective. Per-sequence scores are averaged over the evaluated batch. The official eval path scores batch index 0 inside `get_channel_benchmarks`; averaging over the batch is an extra reduction on top of those helpers.

## Ablation (not in the paper)

`variant = hidden_match` runs the same two passes (teacher-forced and free-running) and does **not** build a discriminator, \(C_d\), \(C_f\), \(C_t\), or the accuracy gate.

The extra term is

\[
\lambda \,\mathrm{mean}\big((h_{\mathrm{free}} - \mathrm{stopgrad}(h_{\mathrm{TF}}))^2\big)
\]

with \(\lambda =\) `hidden_match_weight` (default 1). \(h\) is the post-update GRU state. The mean is over time, batch, and units, i.e. the squared Frobenius norm divided by the number of elements. The paper does not define this penalty or this reduction. The stop-gradient is on the teacher-forced state, so the penalty moves the free-running state toward the teacher-forced state and does not also move the teacher-forced state toward the free-running one. Generator loss is the repo NLL plus this term.

Logs and checkpoints from this variant are labeled `hidden_match`. It is not a Professor Forcing run.
