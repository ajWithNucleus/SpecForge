# Eagle3 Shared Head Training

## Overview

The **Eagle3 Shared Head** variant is an alternative to the standard Eagle3 draft model that addresses vocabulary mapping issues by using the **full target model vocabulary** instead of a reduced draft vocabulary.

## Problem with Standard Eagle3

In standard Eagle3 training, the draft model uses a reduced vocabulary (e.g., 16K tokens) instead of the full target vocabulary (e.g., 128K tokens). This creates two main issues:

### 1. Vocabulary Filtering and Renormalization

```python
# Target produces logits over FULL vocab (128K tokens)
target_logits = [2.1, 0.3, 5.2, ..., 3.4]  # 128K values

# Filter to draft vocab only (16K tokens)
target_logits_filtered = target_logits[t2d]  # Select draft vocab positions

# Renormalize (redistributes probability mass!)
target_p = Softmax(target_logits_filtered)
```

**Problem**: When the target has significant probability mass on tokens outside the draft vocabulary, renormalization redistributes that mass to tokens in the draft vocabulary. This makes the target distribution appear **artificially more concentrated** than it really is.

### 2. Impact on Loss Functions

Both forward KL and reverse KL losses are affected:

**Forward KL (cross-entropy)**:
- Loss = -Σ p_target · log p_draft
- Draft learns to match the artificially concentrated target
- Results in **moderate overconfidence**

**Reverse KL** (worse):
- Loss = Σ p_draft · [log p_draft - log p_target]
- Mode-seeking behavior + artificial concentration
- Results in **high overconfidence**

### Example

```
Original target distribution:
  Token 5 (in draft):     0.40  ← argmax
  Token 10 (in draft):    0.25
  Token 50000 (NOT):      0.30  ← significant mass outside!
  Others (in draft):      0.05

After renormalization to draft vocab:
  Token 5:  0.40/0.70 = 0.571  ← artificially increased
  Token 10: 0.25/0.70 = 0.357  ← artificially increased
  Others:   0.05/0.70 = 0.071
```

The target was uncertain (0.40 confidence), but after filtering appears confident (0.571).

## Solution: Shared Head Model

The **Eagle3 Shared Head** variant completely avoids these problems by:

1. **Using the full target vocabulary** (e.g., 128,256 tokens)
2. **Sharing the lm_head weights** from the target model (frozen during training)
3. **No vocabulary mapping** needed (identity mapping)

### Benefits

✅ **No renormalization artifacts** - Target probabilities are preserved exactly
✅ **Better calibration** - Draft model doesn't become overconfident
✅ **Works with both loss types** - Forward KL and reverse KL are both safe
✅ **Simpler training** - No need to generate vocab mapping files
✅ **Direct comparison** - Logits are in the same space as target

### Trade-offs

⚠️ **Increased memory** - Full vocab (128K) vs reduced (16K) requires ~8x more memory for lm_head
⚠️ **Slower training** - Computing loss over full vocab is slower
⚠️ **Frozen lm_head** - The lm_head cannot be fine-tuned (only the backbone is trained)

## Usage

### 1. Create Config File

Create a config file with `draft_vocab_size` equal to `vocab_size`:

```json
{
  "architectures": [
    "LlamaForCausalLMEagle3SharedHead"
  ],
  "vocab_size": 128256,
  "draft_vocab_size": 128256,
  "hidden_size": 4096,
  "num_hidden_layers": 1,
  ...
}
```

See `configs/llama3-8B-eagle3-shared-head.json` for a complete example.

### 2. Run Training

Use the same training script as standard Eagle3:

```bash
python scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3-shared-head.json \
    --train-data-path data/train.jsonl \
    --output-dir outputs/llama3-8b-eagle3-shared \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --num-epochs 5 \
    --ttt-length 7
```

The training script will automatically:
- Load the embedding from the target model
- Load the lm_head from the target model
- Freeze both embedding and lm_head
- Train only the backbone (fc, midlayer, norm)

### 3. Key Differences

**Standard Eagle3**:
```bash
# Generates vocab mapping from training data
# draft_vocab_size: 16000 (reduced)
# Trains: fc, midlayer, norm, lm_head
```

**Shared Head Eagle3**:
```bash
# No vocab mapping needed (identity mapping)
# draft_vocab_size: 128256 (full)
# Trains: fc, midlayer, norm
# Frozen: embedding, lm_head (shared from target)
```

## Implementation Details

### Model Architecture

The `LlamaForCausalLMEagle3SharedHead` class inherits from the standard Eagle3 model but:

1. **Identity mapping**: `t2d` is all True, `d2t` is [0, 1, 2, ..., vocab_size-1]
2. **Full vocab lm_head**: Same size as target model
3. **Frozen lm_head**: Weights loaded from target and frozen
4. **No vocab mapping file**: `load_vocab_mapping()` is a no-op

### Loss Computation

The loss computation is the same as standard Eagle3, but:

- All positions contribute to loss (no skipping due to vocab mismatch)
- No renormalization artifacts
- Target probabilities are exact (not redistributed)

```python
# Standard Eagle3
target_p = Softmax(target_logits[..., t2d])  # Filter + renormalize

# Shared Head Eagle3
target_p = Softmax(target_logits)  # No filtering, exact probabilities
```

## When to Use

### Use Shared Head When:

✅ You're using **reverse KL loss** (avoids overconfidence issues)
✅ You want **better calibration** of draft model probabilities
✅ You have **enough GPU memory** for full vocab
✅ You want **simpler training** (no vocab mapping)

### Use Standard Eagle3 When:

✅ You need **memory efficiency** (8x less memory for lm_head)
✅ You want **faster training** (smaller vocab = faster loss computation)
✅ You want to **fine-tune the lm_head**
✅ Your draft vocab covers **99%+ of token frequency** (renormalization artifacts are minimal)

## Comparison

| Aspect | Standard Eagle3 | Shared Head Eagle3 |
|--------|----------------|-------------------|
| Vocab size | 16K (reduced) | 128K (full) |
| lm_head params | ~512M (16K × 4K × 8B) | ~4GB (128K × 4K × 8B) |
| Renormalization | Yes (artifacts) | No (exact) |
| Calibration | Moderate overconfidence | Better calibration |
| Reverse KL | Risky (high overconfidence) | Safe (no artifacts) |
| Memory | Lower | Higher (~8x for lm_head) |
| Training speed | Faster | Slower |
| Simplicity | Needs vocab mapping | No vocab mapping |

## Example Training Command

```bash
# Shared Head variant (full vocab)
python scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3-shared-head.json \
    --train-data-path data/sharegpt.jsonl \
    --output-dir outputs/llama3-8b-eagle3-shared \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --num-epochs 3 \
    --ttt-length 7 \
    --attention-backend flex_attention
```

## Recommended Settings

For **reverse KL loss** users:

```bash
# Add this flag (when implemented in eagle3.py)
--loss-type reverse_kl
```

The shared head variant is especially recommended when using reverse KL to avoid the compounding effects of:
1. Vocabulary renormalization (artificial concentration)
2. Mode-seeking behavior (natural concentration)

## FAQs

**Q: Can I convert a trained shared head model to standard Eagle3?**
A: No, they have different vocab sizes in the lm_head.

**Q: Can I fine-tune the lm_head?**
A: No, it's frozen. Only the backbone (fc, midlayer, norm) is trained.

**Q: How much more memory does it use?**
A: Approximately 8x more for the lm_head (128K/16K = 8). For a 4096 hidden size model, that's ~4GB vs ~512MB.

**Q: Is inference different?**
A: No, inference works the same way. The draft model produces logits over the full vocab.

**Q: Should I always use shared head?**
A: Not necessarily. If your draft vocab covers 99%+ of token frequency and you use forward KL, the standard approach works well and is more memory efficient.

## Conclusion

The Eagle3 Shared Head variant provides a cleaner, more principled approach to draft model training by avoiding vocabulary mapping artifacts. While it requires more memory, it offers better calibration and makes reverse KL loss viable - especially important for users who want on-policy distillation.
