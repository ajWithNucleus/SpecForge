# Eagle3 Shared Head Implementation Summary

## What Was Implemented

A new **Eagle3 Shared Head** variant that completely avoids vocabulary mapping/renormalization issues by using the full target model vocabulary.

## Key Files Modified/Created

### 1. Model Implementation
- **`specforge/modeling/draft/llama3_eagle.py`**
  - Added `LlamaForCausalLMEagle3SharedHead` class (lines 1354-1507)
  - Implements `load_lm_head()` and `freeze_lm_head()` methods
  - Uses identity mapping for t2d/d2t (no vocab filtering)

### 2. Factory Support
- **`specforge/modeling/auto.py`** (already configured)
  - `_SHARED_HEAD_MODEL_MAPPING` supports the new architecture
  - `AutoDraftModelConfig._config_mapping` includes shared head config

### 3. Training Script
- **`scripts/train_eagle3.py`** (line 385-389)
  - Automatically detects shared head models via `hasattr(draft_model, 'load_lm_head')`
  - Loads and freezes lm_head from target model

### 4. Configuration
- **`configs/llama3-8B-eagle3-shared-head.json`** (new file)
  - Example config with `draft_vocab_size = vocab_size`
  - Architecture: `"LlamaForCausalLMEagle3SharedHead"`

### 5. Documentation
- **`docs/shared_head_training.md`** (new file)
  - Comprehensive guide explaining the problem and solution
  - Usage instructions and examples
  - Comparison table and recommendations

## How It Works

### Standard Eagle3 (with vocab filtering)
```python
# 1. Target produces logits over 128K tokens
target_logits = target_model(...)  # [B, T, 128256]

# 2. Filter to draft vocab (16K tokens)
target_logits_filtered = target_logits[..., t2d]  # [B, T, 16000]

# 3. Renormalize (PROBLEM: redistributes probability mass!)
target_p = Softmax(target_logits_filtered)

# 4. Draft predicts over 16K tokens
draft_logits = draft_model(...)  # [B, T, 16000]

# 5. Compute loss
loss = KL(draft_logits, target_p)
```

### Shared Head Eagle3 (no vocab filtering)
```python
# 1. Target produces logits over 128K tokens
target_logits = target_model(...)  # [B, T, 128256]

# 2. No filtering - use directly
target_p = Softmax(target_logits)  # [B, T, 128256]

# 3. Draft uses SAME lm_head as target (frozen)
draft_logits = draft_model(...)  # [B, T, 128256]

# 4. Compute loss (no renormalization artifacts!)
loss = KL(draft_logits, target_p)
```

## Usage Example

```bash
# 1. Use the shared head config
python scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3-shared-head.json \
    --train-data-path data/train.jsonl \
    --output-dir outputs/llama3-8b-shared-head \
    --batch-size 4 \
    --num-epochs 3 \
    --ttt-length 7

# The script will automatically:
# - Load embedding from target (frozen)
# - Load lm_head from target (frozen)
# - Train only: fc, midlayer, norm
```

## Key Benefits

✅ **No renormalization artifacts** - Target probabilities preserved exactly
✅ **Better calibration** - Draft doesn't become overconfident
✅ **Safe for reverse KL** - Avoids compounding concentration issues
✅ **Simpler** - No vocab mapping file generation
✅ **All positions contribute** - No skipped positions due to vocab mismatch

## Trade-offs

⚠️ **More memory** - Full vocab lm_head (~8x larger: 4GB vs 512MB)
⚠️ **Slower training** - Loss computed over 128K tokens instead of 16K
⚠️ **Frozen lm_head** - Cannot fine-tune the lm_head weights

## When to Use

**Use Shared Head when:**
- Using reverse KL loss (avoids overconfidence)
- Want better calibration
- Have enough GPU memory
- Prefer simpler training pipeline

**Use Standard Eagle3 when:**
- Memory constrained
- Want faster training
- Draft vocab covers 99%+ of token frequency
- Want to fine-tune lm_head

## Testing

To verify the implementation works:

```bash
# 1. Create a small test dataset
echo '{"conversations": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}' > test.jsonl

# 2. Run training for 1 step
python scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3-shared-head.json \
    --train-data-path test.jsonl \
    --output-dir test_output \
    --batch-size 1 \
    --num-epochs 1 \
    --max-num-steps 1

# 3. Check logs for:
# - "Loading and freezing lm_head from target model (shared head variant)"
# - Verify lm_head.weight.requires_grad = False
# - Training completes without vocab mapping errors
```

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│ Eagle3 Shared Head Draft Model                 │
├─────────────────────────────────────────────────┤
│ embed_tokens (from target, frozen)              │
│ ↓                                               │
│ fc (trainable) - Projects 3×hidden to hidden    │
│ ↓                                               │
│ midlayer (trainable) - 1 transformer layer      │
│ ↓                                               │
│ norm (trainable) - RMSNorm                      │
│ ↓                                               │
│ lm_head (from target, FROZEN) - Full vocab      │
│ ↓                                               │
│ logits [B, T, 128256]                           │
└─────────────────────────────────────────────────┘
```

## Summary

This implementation provides a clean solution to the vocab mapping/renormalization problem by simply using the full target vocabulary and sharing the lm_head weights. While it requires more memory, it offers better training dynamics and makes reverse KL loss a viable option without overconfidence issues.

The implementation is backward compatible - existing standard Eagle3 training continues to work as before. Users can choose the variant that best fits their memory constraints and training objectives.
