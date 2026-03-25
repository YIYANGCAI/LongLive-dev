# Reference Image Control Integration Analysis

## Codebase Flow: Summary

The training call chain works as follows:

### `fwdbwd_one_step_streaming` (trainer/distillation.py:1056)
- Fetches a text prompt batch → encodes via `text_encoder` → produces `conditional_dict = {"prompt_embeds": Tensor[B, 512, T5_dim]}`
- Calls `streaming_model.generate_next_chunk()`

### `StreamingTrainingModel.generate_next_chunk` (model/streaming_training.py:407)
- Manages sequence state: tracks `current_length`, `previous_frames` (last 21 latent frames), and the `conditional_info` dict
- Samples noise for new frames, calls `_generate_chunk()`

### `StreamingTrainingModel._generate_chunk` (model/streaming_training.py:189)
- Determines the active `conditional_dict` (handles prompt-switch logic for DMDSwitch)
- Delegates to `inference_pipeline.generate_chunk_with_cache(**kwargs)`

### `StreamingSwitchTrainingPipeline.generate_chunk_with_cache` (pipeline/streaming_switch_training.py:35)
- Iterates block-by-block, running a multi-step denoising loop
- On hitting `switch_frame_index`: calls `_recache_after_switch()`, which:
  1. Zeros the self-attention KV cache (`kv_cache1`)
  2. Zeros the cross-attention cache (`crossattn_cache`)
  3. Re-runs the last ≤21 latent frames through the generator under the **new** `conditional_dict` to re-seed the caches under the new conditioning
- Continues generation using `switch_conditional_dict` for subsequent blocks

---

## Key Data Structures

| Structure | Shape | Role |
|---|---|---|
| `conditional_dict["prompt_embeds"]` | `[B, 512, T5_dim]` | Text conditioning tokens, fed as `context` to cross-attention |
| `kv_cache1` | list of 30 blocks × `{k,v: [B, kv_size, 12, 128]}` | Self-attention KV cache — accumulates video frame tokens across chunks |
| `crossattn_cache` | list of 30 blocks × `{k,v: [B, 512, 12, 128]}` | Cross-attention cache — caches K,V projections of text tokens, lazily initialized on first use |

The model forward in `WanDiffusionWrapper.forward()` (utils/wan_wrapper.py:280) passes `prompt_embeds` as `context` into `CausalWanModel`, where it is consumed in each transformer block's **cross-attention** layer.

---

## Analysis: Integrating Reference Image Control (Idea 1 — ID Memory Bank)

### 1. VAE Encoding is Already in Place

`WanVAEWrapper.encode_to_latent()` (utils/wan_wrapper.py:80) already handles exactly what is described in Idea 1:
- Input: pixel tensor `[B, C, T, H, W]`
- Output: latent tensor `[B, T, 16, 60, 104]`

For a single reference image, treat it as T=1, producing a latent of shape `[B, 1, 16, 60, 104]`, which patchifies to **1560 tokens** per image — the same token density as one video frame.

This is already how `initial_latent` works for i2v mode in `setup_sequence()` (model/streaming_training.py:350).

---

### 2. The Prompt Switch Mechanism is the Direct Analogy

The existing prompt-switch system gives an exact structural template:

| Prompt Switching (existing) | Reference Image Control (proposed) |
|---|---|
| New text prompt → `switch_conditional_dict` | New reference image → `id_conditional_dict` |
| `crossattn_cache` reset + rebuilt under new text | ID cache seeded with reference image tokens |
| Switch happens once at `switch_frame_index` | Switch/update can happen at any user-defined frame |
| Text K,V stored in `crossattn_cache` [B, 512, 12, 128] | Image K,V stored in `id_cache` [B, 1560, 12, 128] |

---

### 3. Two Integration Paths

#### Path A — Cross-attention Injection (lighter-weight)

Extend `conditional_dict` to include image embeddings alongside text:

```python
conditional_dict = {
    "prompt_embeds":  Tensor[B, 512, dim],   # text tokens (existing)
    "ref_img_embeds": Tensor[B, N_ref, dim], # reference image tokens (new)
}
```

The cross-attention in each transformer block would attend to both. This requires:
- Modifying the cross-attention layers in `wan/modules/causal_model.py` to handle a second context source
- Extending `crossattn_cache` from 512 to `512 + N_ref` slots

#### Path B — Self-attention Token Concatenation (as described in Idea 1)

This is closer to the paper's description and more expressive:

1. Encode the reference image via VAE → latent `[B, 1, 16, 60, 104]`
2. Run it through the generator (no noise, timestep=0) to compute K,V projections at all 30 transformer blocks → store these as the **ID Memory Bank** (a new cache structure parallel to `kv_cache1`)
3. At each denoising step, video token queries attend to both: the rolling video KV cache **and** the ID Memory Bank
4. When a new reference is added: compute its K,V projections and **append** to the ID Memory Bank (do not replace existing entries)

This is structurally identical to how `_initialize_kv_cache()` works, but the ID cache is seeded from reference images rather than generated video frames, and it persists without being evicted by the sliding window.

---

### 4. Where the ID Memory Bank Lives and Updates

The closest existing structure is `crossattn_cache` — initialized once and lazily populated at the first generator forward call. The ID Memory Bank would follow the same pattern but with key differences:

- **Seeded explicitly** by the user providing a reference image (via `setup_sequence()` or a new `add_reference()` call on `StreamingTrainingModel`)
- **Sized** `[B, N_ref × 1560, 12, 128]` per block, where `N_ref` grows as more references are added
- **Persistent** across all chunks — never evicted by the sliding-window rolling logic that manages `kv_cache1`

The `_recache_after_switch()` method (pipeline/streaming_switch_training.py:244) shows exactly how to force a re-seeding: run frames through the generator under a new condition, then zero the cross-attention cache to flush old conditioning. Analogously, when a new identity is introduced mid-video, the ID Memory Bank is updated by appending new identity K,V projections, while the existing video KV cache for already-generated frames remains untouched.

---

### 5. Files to Touch in a Future Implementation

| File | Change Needed |
|---|---|
| `utils/dataset.py` | Supply reference images alongside text prompts in each batch |
| `model/streaming_training.py` | Extend `setup_sequence()` and `state` dict to hold reference latents / ID Memory Bank; add `add_reference()` method |
| `model/streaming_training.py` `_generate_chunk` | Pass ID Memory Bank down to the pipeline |
| `pipeline/streaming_training.py` `generate_chunk_with_cache` | Add `id_cache` parameter; inject it into the generator forward call |
| `utils/wan_wrapper.py` `WanDiffusionWrapper.forward` | Accept `id_cache` and pass through to the underlying model |
| `wan/modules/causal_model.py` `CausalWanSelfAttention.forward` | Merge `id_cache` K,V with `kv_cache1` K,V before the `flex_attention` call |

### 6. The Critical Attention-Level Change

The core modification is in `CausalWanSelfAttention.forward` (wan/modules/causal_model.py). Currently, when `kv_cache` is provided, attention is computed as:

```
Q (video) attends to → K,V from kv_cache1 (rolling video history)
```

With the ID Memory Bank, this becomes:

```
Q (video) attends to → [K,V from kv_cache1 (video history)] ++ [K,V from id_cache (identity tokens)]
```

Concretely, before calling `flex_attention`, `temp_k` and `temp_v` (which already hold the rolling video KV) would be extended along the sequence dimension by appending the identity cache's K,V tensors. This lets video token queries retrieve identity features unconditionally at every denoising step and every chunk, without the identity tokens being displaced by the sliding-window eviction logic.
