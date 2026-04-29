**Diagnosis of your results:**

Looking at the classification report carefully:

| Class             | Precision | Recall   | Problem                               |
| ----------------- | --------- | -------- | ------------------------------------- |
| Machine1_Normal   | 1.00      | 0.84     | Fine                                  |
| Machine1_Abnormal | 0.56      | **1.00** | Predicts abnormal for everything      |
| Machine2_Normal   | 0.88      | 0.58     | Confused with Abnormal                |
| Machine2_Abnormal | 0.22      | **0.60** | Low precision = many false positives  |
| Machine3_Normal   | 0.98      | **0.22** | Model almost never predicts this      |
| Machine3_Abnormal | 0.21      | **0.98** | Model almost always predicts Abnormal |

The core problem is **not just imbalance** — it's that your model is collapsing the Normal/Abnormal distinction within Machine2 and Machine3. Machine1 works fine (~5:1 ratio), but Machine2 and Machine3 have similar ratios yet fail completely. This means the model learned Machine1's acoustic signature but failed to learn discriminative features for M2/M3.

**Root causes (research-backed):**

1. **Catastrophic forgetting between machines**: The 6-class flat softmax treats inter-machine and intra-machine distinctions identically. The model finds Machine1 easy and over-optimizes for it. _Reference: Rebuffi et al., "Learning multiple visual domains with residual adapters" (NeurIPS 2017)._

2. **The 1/√count weighting is still insufficient**: Machine1_Normal has ~2430 samples; Machine3_Abnormal has ~453 — a 5.4× ratio. `1/sqrt(2430)` vs `1/sqrt(453)` gives only 2.3× upweighting, which is too weak when the model can hide errors in the dominant class. _Reference: King & Zeng, "Logistic Regression in Rare Events Data" (2001); Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)._

3. **SpecAugment is hurting minority classes**: Masking 3 frequency bands × up to 40 rows on a 128-row mel-spec = masking up to 94% of frequency content. For minority classes with fewer examples, this destroys too much signal. _Reference: Park et al., "SpecAugment" (Interspeech 2019) — original paper used max_freq_mask=27 on 80-mel, i.e. ~34% of bins._

4. **Overfitting despite dropout**: val_loss explodes from epoch 2 onwards (0.59 → 2.91 → 3.75) while train_acc → 99.4%. Dropout at test time is inactive. You need the model to be less confident, not just regularized. _Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)._

5. **Label smoothing missing**: CrossEntropyLoss with hard labels + class weights pushes the model to be overconfident on easy classes. _Reference: Müller et al., "When Does Label Smoothing Help?" (NeurIPS 2019)._

**Specific fixes I'm making:**

1. **Effective Number of Samples weighting** (Cui et al. CVPR 2019): `w = (1 - β) / (1 - β^n)` with β=0.9999. This is theoretically grounded, not heuristic.

2. **Label smoothing = 0.1** in CrossEntropyLoss — prevents overconfidence on easy classes.

3. **SpecAugment parameters reduced to safe values**: `freq_mask=27, time_mask=20, n_freq=2, n_time=2` — matching the original paper's proportions.

4. **Mixup augmentation at α=0.3**: Interpolates between samples of different classes, forcing the model to learn smooth decision boundaries between Machine2/3 Normal vs Abnormal. _Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)._

5. **Reduce EPOCHS to 12 with early stopping patience=4**: Val loss is diverging after epoch 7 — more epochs only deepens overfitting. Save the best checkpoint by val loss, not val accuracy.

6. **Warmup scheduler + cosine**: Instead of cosine-only, use a 2-epoch linear warmup then cosine, which prevents the model from memorizing easy classes in early epochs. *Reference: Goyal et al., "Accurate, Large Minibatch SGD" (2017).*Here's the complete breakdown of what I changed and the exact research behind each decision:

---

## What I changed and why (research-backed)

### [FIX 1] Effective Number of Samples weighting — replaces `1/√n`

Your `1/sqrt(label_counts)` gives only a 2.3× boost to minority classes vs. the majority. **Cui et al. CVPR 2019** showed that as `n` grows, each additional sample overlaps more with existing ones — the "effective" new information shrinks. The formula `(1 − β^n) / (1 − β)` with `β=0.9999` captures this saturation, producing a theoretically grounded weight that doesn't under-boost small classes at large `n`. **Paper: https://arxiv.org/abs/1901.05555**

### [FIX 2] Label smoothing ε = 0.1

Your model was pushing Machine1 logits to near-certainty (precision=1.00), which crowds out gradient signal for M2/M3. Label smoothing prevents any class from receiving a target probability of 1.0, forcing the model to stay uncertain enough to keep learning M2/M3. **Müller et al. NeurIPS 2019: https://arxiv.org/abs/1906.02629**

### [FIX 3] SpecAugment parameters calibrated to the original paper

Your v1 used `freq_mask=40, n_freq=3` on 128 mel bins — that's up to 93% of frequency content destroyed per sample. The **Park et al. Interspeech 2019** paper used `F=27` on 80-mel bins (~34%). For minority classes with few examples, destroying 93% of frequency information makes each sample near-useless. I reduced to `freq_mask=27, n_freq=2` (~42% max). **Paper: https://arxiv.org/abs/1904.08779**

### [FIX 4] Mixup augmentation α = 0.3

This is the key fix for M2/M3 boundary collapse. Mixup interpolates between pairs of samples — including M2_Normal+M2_Abnormal and M3_Normal+M3_Abnormal — forcing the model to learn a smooth decision boundary between them rather than collapsing everything to "Abnormal." **Zhang et al. ICLR 2018: https://arxiv.org/abs/1710.09412**

### [FIX 5] Linear warmup (2 epochs) + cosine annealing

At epoch 1 with full LR, the model memorised Machine1 (the easiest class) before it had seen enough M2/M3 examples to learn from. 2-epoch linear warmup gives the model time to see all class distributions before committing to high-LR gradient steps. **Goyal et al. 2017: https://arxiv.org/abs/1706.02677**

### [FIX 6] Early stopping on val_loss (patience=4), save by val_loss

Your val_loss was `0.59 → 2.91 → 3.75` by the end — the model at epoch 7 was the last useful one. Saving by val_accuracy masked this because accuracy can stay stable while the model becomes increasingly miscalibrated. **Prechelt, "Early Stopping — But When?" Springer 1998.**

### [FIX 7] Dropout back to 0.5

Dropout 0.6 + SpecAugment + Mixup = three independent regularisers stacked. **Srivastava et al. JMLR 2014** showed 0.5 is the empirical optimum for FC layers. Over-regularisation prevents the model from learning M2/M3 features at all. **Paper: https://jmlr.org/papers/v15/srivastava14a.html**
