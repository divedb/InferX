# FlashInfer `SinglePrefillWithKVCacheDevice` 实现详解

> 源文件: `third_party/flashinfer/include/flashinfer/attention/prefill.cuh:1333`
> 算法: FlashAttention-2 (Dao 2023), 非 Hopper (sm_80~sm_89) 路径

---

## 1. 算法概述

### 1.1 要计算什么

单个请求的 scaled-dot-product attention:

```
S = Q @ K^T * scale          // [qo_len, kv_len]
P = softmax(S, dim=-1)        // [qo_len, kv_len]
O = P @ V                     // [qo_len, head_dim]
```

其中 Q `[qo_len, num_qo_heads, head_dim]`，K/V `[kv_len, num_kv_heads, head_dim]`。支持 GQA (`num_qo_heads > num_kv_heads`) 和 causal mask。

### 1.2 为什么不能直接做

朴素实现的中间矩阵 S 和 P 都是 `[qo_len, kv_len]`，当序列长时 (例如 4K × 4K) 远超 GPU 显存。而且两次大矩阵乘法的中间结果要写回 HBM 再读回来，带宽浪费严重。

### 1.3 FlashAttention 的核心思想

**分块 (Tiling)** + **在线 softmax (Online Softmax)**:

1. 把 Q 沿行方向切成 CTA_TILE_Q 大小的块，K/V 沿行方向切成 CTA_TILE_KV 大小的块。
2. 每个线程块 (CTA) 只加载一个 Q 块到 shared memory (常驻)，然后流式地遍历 K/V 块。
3. 每处理一个 K/V 块就立即算出局部 S 和 P，用 P 更新 O —— **中间矩阵 S/P 永远不离开寄存器**。
4. 在线 softmax 保证跨块拼接时结果正确：维护行最大值 m 和行求和 d，每来一个新块就修正之前的累加。

这样整个 attention 只需要 `O(CTA_TILE_Q × head_dim + CTA_TILE_KV × head_dim)` 的 shared memory，与序列长度无关。

---

## 2. 线程与内存层次

### 2.1 Grid 维度

```
gridDim = (ceil_div(qo_len * group_size, CTA_TILE_Q),  num_chunks,  num_kv_heads)
            blockIdx.x                                        blockIdx.y   blockIdx.z
```

| 维度 | 含义 |
|---|---|
| `blockIdx.x` | Q-tile 索引。`group_size = num_qo_heads / num_kv_heads`，把 GQA 的多个 Q head "打包"进同一个 tile |
| `blockIdx.y` | KV chunk 索引 (split-KV)。当并行度不足时把 kv_len 切成多段，最后用 `MergeStates` 合并 |
| `blockIdx.z` | KV head 索引。一个 KV head 对应 `group_size` 个 Q heads |

### 2.2 Block 维度 (Warp 网格)

```
blockDim = (32,  NUM_WARPS_Q,  NUM_WARPS_KV)
            threadIdx.x  threadIdx.y  threadIdx.z
```

总共 `NUM_WARPS_Q * NUM_WARPS_KV` 个 warp (通常 4 个)，它们组成一个 2D 网格:

```
                   N轴(kv_len方向)
                   ←─────────────→
              ┌────┬────┬────┬────┐
   M轴      Q │ W0 │ W1 │ W2 │ W3 │   NUM_WARPS_KV = 4, NUM_WARPS_Q = 1
 (Q行方向)    └────┴────┴────┴────┘   (CTA_TILE_Q = 16 或 64 时)
               
              ┌────┐
              │ W0 │   NUM_WARPS_Q = 4, NUM_WARPS_KV = 1
              ├────┤   (CTA_TILE_Q = 128 或 64 时)
              │ W1 │
              ├────┤
              │ W2 │
              ├────┤
              │ W3 │
              └────┘
```

- **`threadIdx.y` (Q 维 warp)**: 每个 warp 负责 `NUM_MMA_Q` 个 16-row fragment，共 `NUM_MMA_Q × 16` 行 Q。
- **`threadIdx.z` (KV 维 warp)**: 每个 warp 负责 `NUM_MMA_KV` 个 16-row fragment。

当 `NUM_WARPS_KV > 1` 时，不同 KV-warp 各自处理一段 KV 列，最后通过 shared memory 做跨 warp 规约 (`threadblock_sync_mdo_states`)。

### 2.3 Shared Memory 布局

```
SharedStorageQKVO {
    DTypeQ  q_smem[CTA_TILE_Q × HEAD_DIM_QK]    // 常驻，加载一次
    DTypeKV k_smem[CTA_TILE_KV × HEAD_DIM_QK]   // 流式，每次迭代换一块
    DTypeKV v_smem[CTA_TILE_KV × HEAD_DIM_VO]   // 流式
    float   cta_sync_o_smem[...]                // 跨 warp 规约用
    float2  cta_sync_md_smem[...]
}
```

K 和 V 各只有一块 smem buffer，通过 `cp.async` 的 commit/wait group 做隐式双缓冲 (见第 5 节)。

smem 使用 **swizzle 布局** (`get_permuted_offset`) 避免 bank conflict: 地址 = `row * stride + (col ^ (row % 8))`，让不同行同一列的数据落到不同 bank。

---

## 3. 寄存器 Fragment 布局

每个 warp 持有以下寄存器数组 (这是整个算法的核心数据结构):

```cpp
// S = QK^T 的输出 tile: [M轴 NUM_MMA_Q] × [N轴 NUM_MMA_KV]
DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];

// O = PV 的输出 tile: [M轴 NUM_MMA_Q] × [N轴 NUM_MMA_D_VO]
float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];

// 在线 softmax 状态: 每行一个 m (max) 和一个 d (sum)
DTypeQKAccum m[NUM_MMA_Q][2];   // [2] 对应 16 行中的前 8 行和后 8 行
float        d[NUM_MMA_Q][2];
```

### 3.1 为什么是 `[8]`?

`mma.sync m16n16k16` 指令 (fp16 输入, fp32 累加) 的输出 fragment 对每个线程持有 8 个 fp32 寄存器。一个 warp (32 线程) 共同描述一个 16×16 的输出 tile，布局如下 (以 `s_frag[mma_q][mma_kv]` 为例):

```
8 个寄存器在每个 lane 中的行/列映射:
  reg_id:   0   1   2   3   4   5   6   7
  ──────────────────────────────────────────
  行 (i):   0   0   8   8   0   0   8   8     (lane % 16 决定列, 行偏移 0/8)
  列 (j):   0   1   0   1   8   9   8   9     (j 组 = lane / 16, j 偏移 0 或 8)
```

所以 `m[NUM_MMA_Q][2]` 里的 `[2]` 对应一个 16-row fragment 的前 8 行 (`j=0`) 和后 8 行 (`j=1`)。`s_frag[..][..][j*2+0]` 和 `s_frag[..][..][j*2+1]` 属于同一行的两个列位置，`s_frag[..][..][j*2+4]` 和 `s_frag[..][..][j*2+5]` 属于另一组列。

### 3.2 Fragment 维度速查

| Fragment | 第一维 | 第二维 | 每元素含义 |
|---|---|---|---|
| `s_frag` | `NUM_MMA_Q` (M: Q行) | `NUM_MMA_KV` (N: KV列) | QK^T 的部分积 |
| `o_frag` | `NUM_MMA_Q` (M: Q行) | `NUM_MMA_D_VO` (N: head_dim列) | PV 的累加输出 |
| `m` | `NUM_MMA_Q` | 2 (前8行/后8行) | 每行已见最大 logit |
| `d` | `NUM_MMA_Q` | 2 | 每行 softmax 分母的累加 |

---

## 4. 主循环逐步解析

### 4.1 初始化 (`prefill.cuh:1399-1409`)

```cpp
init_states(variant, o_frag, m, d);
```

- `o_frag` 全清零。
- `m` 初始化为 `-inf` (还没见过任何 logit)。
- `d` 初始化为 `1.0` (乘法单位元，后面会乘以 rescale 因子)。

### 4.2 加载 Q tile (`load_q_global_smem`, `prefill.cuh:428`)

```cpp
load_q_global_smem(qo_packed_idx_base, qo_len, q_ptr_base, ...);
cp_async::commit_group();
```

- 只由 `threadIdx.z == 0` 的 warp 执行 (Q 只需要加载一份，不需要 KV 维的 warp 重复)。
- 用 `cp.async` (128-bit 粒度) 从 gmem 异步预取到 smem。
- Q tile 一旦加载就**常驻 smem**，整个 kernel 期间不换。
- 如果启用 RoPE，等 Q 加载完后 `q_smem_inplace_apply_rotary` 在 smem 上原地做旋转位置编码。

### 4.3 预取第一块 K 和 V (`prefill.cuh:1479-1484`)

```cpp
produce_kv<false, kNoFill>(k_smem, ...);  cp_async::commit_group();  // group 1: K tile 0
produce_kv<true,  kFillZero>(v_smem, ...); cp_async::commit_group();  // group 2: V tile 0
```

- V 用 `kFillZero` 模式：如果 kv_idx 超出范围，smem 填零 (避免越界读 + mask 自然为 0)。
- 此时 pipeline 里有 2 个未完成的 commit group。

### 4.4 KV 主循环 (`prefill.cuh:1486-1527`)

```cpp
for (iter = 0; iter < num_iterations; ++iter) {
    // ── (A) 等 K tile 就绪 ──
    cp_async::wait_group<1>();   // 等 group 1 (K) 完成，pipeline 里剩 group 2 (V)
    block.sync();

    // ── (B) S = Q @ K^T ──
    compute_qk(&qo_smem, &k_smem, s_frag);

    // ── (C) logits 变换 + mask ──
    logits_transform(...);   // scale, ALiBi bias, soft_cap
    if (need_mask) logits_mask(...);  // causal / sliding window

    // ── (D) 在线 softmax 更新 ──
    update_mdo_states(variant, s_frag, o_frag, m, d);

    // ── (E) 预取下一块 K ──
    block.sync();
    produce_kv<false>(k_smem, ..., (iter+1) * CTA_TILE_KV, ...);
    cp_async::commit_group();  // group 3: K tile (iter+1)

    // ── (F) 等 V tile 就绪 ──
    cp_async::wait_group<1>();  // 等 group 2 (V) 完成
    block.sync();

    // ── (G) O += softmax(S) @ V ──
    compute_sfm_v(&v_smem, s_frag, o_frag, d);

    // ── (H) 预取下一块 V ──
    block.sync();
    produce_kv<true>(v_smem, ..., (iter+1) * CTA_TILE_KV, ...);
    cp_async::commit_group();  // group 4: V tile (iter+1)
}
cp_async::wait_group<0>();  // 收尾：等所有 group 完成
```

#### 4.4.1 pipeline 时序图

`wait_group<N>` 表示"等待直到最多还有 N 个未完成 group"。

```
时间 →
iter 0:  [wait K0] compute_qk → update_mdo → [issue K1] [wait V0] compute_sfm_v → [issue V1]
iter 1:  [wait K1] compute_qk → update_mdo → [issue K2] [wait V1] compute_sfm_v → [issue V2]
...
```

K 和 V 的加载与计算重叠：当 `compute_sfm_v` 在算第 `iter` 块的 V 时，第 `iter+1` 块的 K 已经在通过 `cp.async` 从 gmem 传输。

### 4.5 `compute_qk` — S = Q @ K^T (`prefill.cuh:615`)

三层循环，对应 MMA 的 M/N/K 三个轴:

```cpp
for (mma_d = 0; mma_d < NUM_MMA_D_QK; mma_d++) {     // K轴(收缩), head_dim/16 次
    // 加载 Q fragment (a_frag), 每个 mma_q 一份
    for (mma_q) q_smem->ldmatrix_m8n8x4(..., a_frag[mma_q]);

    // 加载 K fragment (b_frag), 每个 mma_kv 一份
    for (mma_kv) {
        k_smem->ldmatrix_m8n8x4(..., b_frag);
        // 发射 MMA: s_frag += a_frag × b_frag
        for (mma_q) {
            if (mma_d == 0)  mma_init(s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
            else             mma_acc (s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
        }
    }
}
```

**关键**: `mma_d` 是最外层循环。第一次 (`mma_d==0`) 用 `kInit` 模式初始化 `s_frag`，后续累加。这对应 `Q[16行, 16列] × K^T[16列, 16行] → S[16行, 16行]` 的分块矩阵乘——head_dim 被切成 `NUM_MMA_D_QK` 段，每段 16 个元素。

`ldmatrix_m8n8x4` 是 PTX `ldmatrix` 指令的封装：一个 warp 的 32 个线程协作从 smem 加载 4 个 8×8 矩阵，直接排布成 MMA 指令所需的 fragment 格式，无需额外 shuffle。

### 4.6 `update_mdo_states` — 在线 softmax (`prefill.cuh:862`)

这是 FlashAttention 的数学核心。对每个 Q 行 (16-row fragment 分成前 8 行 `j=0` 和后 8 行 `j=1`):

```
1. 找当前 KV 块内的局部最大值 m_local
2. m_new = max(m_prev, m_local)                     // 跨 NUM_MMA_KV 取 max
3. 跨 warp 内 4 个 lane 做 warp shuffle reduce:
   m_new = max(m_new, shfl_xor(m_new, 0x2))         // lane 0↔2, 1↔3
   m_new = max(m_new, shfl_xor(m_new, 0x1))         // lane 0↔1, 2↔3
   → 现在 4 个 lane 持有相同的 m_new (对应同一行的 4 个列段)
4. o_scale = exp2(m_prev * sm_scale - m_new * sm_scale) = exp2((m_prev - m_new) * sm_scale)
5. d *= o_scale                                      // 修正之前的分母
6. o_frag[*][*] *= o_scale                           // 修正之前的 O 累加
7. s_frag = exp2(s_frag * sm_scale - m_new * sm_scale)  // P = softmax 的分子
```

#### 为什么用 exp2 而不是 exp?

FlashInfer 用 `sm_scale_log2 = sm_scale × log2(e)` 把 softmax scale 折叠进 exp2，因为 GPU 的 `ex2.approx` PTX 指令比 `exp` 快得多 (硬件直接支持)。

#### 为什么是 `m × sm_scale` 而不是先 scale 再取 max?

看 `finalize_m` (`prefill.cuh:1035`): m 在整个循环里**不乘 sm_scale**，只在最后才乘。这样 `m` 始终是未缩放的原始 logit 的最大值，而 `s_frag * sm_scale - m * sm_scale` 里 sm_scale 被提出来了——等价于 `exp2((s_frag - m) * sm_scale_log2)`。

#### shuffle reduce 解释

一个 16-row fragment 内，同一行的 16 个元素被 warp 里的 4 个 lane 各持有 4 个 (对应 4 个 mma_kv 列段)。`shfl_xor_sync` 在 lane `0↔2` 和 `1↔3` 之间交换，让 4 个 lane 都得到全局 m_new。这样后续对 `o_frag` 和 `d` 的 rescale 在 4 个 lane 上同步，不会发散。

### 4.7 `compute_sfm_v` — O += P @ V (`prefill.cuh:956`)

```cpp
// 先把 s_frag (float/fp32) 转成 fp16 (MMA 需要 fp16×fp16→fp32)
for (mma_q, mma_kv) vec_cast<DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv], s_frag[...]);

// 行求和 (用于 d): d += sum(P)
for (mma_q, mma_kv) mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv]);

// O += P @ V
for (mma_kv = 0; mma_kv < NUM_MMA_KV; mma_kv++) {      // 收缩轴
    for (mma_d = 0; mma_d < NUM_MMA_D_VO; mma_d++) {    // N轴 (head_dim_vo)
        v_smem->ldmatrix_m8n8x4_trans(..., b_frag);     // V 需要 transpose 加载
        for (mma_q) {
            mma_sync(o_frag[mma_q][mma_d], s_frag_f16[mma_q][mma_kv], b_frag);
        }
    }
}
```

注意循环顺序和 `compute_qk` 不同:
- QK^T 里收缩轴 (head_dim_qk) 在最外层，因为 Q/K fragment 要复用。
- PV 里收缩轴 (kv_len / NUM_MMA_KV) 在最外层，因为 V fragment 每次换。

`ldmatrix_m8n8x4_trans` 加载 V 的转置——因为 V 在 smem 里按行存 (kv_len × head_dim)，但 PV 要求 V 按 `[head_dim, kv_len]` 用，所以 ldmatrix 的 `.trans` 变体在加载时做转置。

### 4.8 `finalize_m` (`prefill.cuh:1035`)

```cpp
if (m[mma_q][j] != -inf)  m[mma_q][j] *= sm_scale_log2;
```

把 m 乘上 scale。如果某行全被 mask 掉 (m == -inf)，不乘——保持 -inf，后续输出该行为 0。

### 4.9 `threadblock_sync_mdo_states` (`prefill.cuh:1091`)

当 `NUM_WARPS_KV > 1` 时，不同 KV-warp 各自维护独立的 m/d/o_frag。循环结束后需要沿 KV 维做一次规约:

1. 所有 warp 把 o_frag 和 (m, d) 写到 shared memory。
2. `__syncthreads()`。
3. 每个 warp 读回所有 KV-warp 的 (m, d)，做**跨 chunk 的在线 softmax 合并**:

```
m_new = max(m_i for i in warps)
d_new = Σ d_i × exp2(m_i - m_new)
o_frag_new = Σ o_frag_i × exp2(m_i - m_new)
```

这和 `update_mdo_states` 里的 rescale 公式完全一致——本质上是在做"把多个独立 softmax 分块的结果合并成一个全局 softmax"。

### 4.10 `transform_output` (`prefill.cuh:1051`)

对每个寄存器元素调用 `variant.OutputTransform`:

```
o_frag[reg] = o_frag[reg] / d    // 归一化: 除以 softmax 分母
```

这是最后一步：把未归一化的加权 V 累加值变成真正的 attention 输出。

### 4.11 `write_o_reg_gmem` (`prefill.cuh:1220`)

用 `stmatrix` 把 `o_frag` 从寄存器写到 smem (作为中转)，再用 128-bit store 从 smem 写到 gmem 的输出张量 O。

同时写 LSE (log-sum-exp) 供 split-KV 合并使用:
```
lse = log2(d) + m * sm_scale_log2
```

---

## 5. cp.async 双缓冲流水线

整个 KV 循环的数据搬运用 `cp.async` + commit/wait group 实现:

```
pipeline 状态 (group 编号是 commit 的顺序):

iter 0 开始时:  pending = {K0(g1), V0(g2)}
  wait_group<1>  → 等 K0 完成 (pending ≤ 1 → 等 g1)
  compute_qk(K0)
  issue K1(g3)   → pending = {V0(g2), K1(g3)}
  wait_group<1>  → 等 V0 完成
  compute_sfm_v(V0)
  issue V1(g4)

iter 1 开始时:  pending = {K1(g3), V1(g4)}
  ...同上...
```

`wait_group<N>` 的语义是"阻塞直到未完成的 group 数 ≤ N"。所以:
- `wait_group<1>` 后启动新 K 加载 = 保证当前 K 已就绪，同时允许 V 还在飞
- `wait_group<0>` (循环结束后) = 等所有搬运完成

这让 gmem→smem 的搬运 (约 ~200 cycles 延迟) 与 smem→reg 的 MMA 计算 (约 ~8 cycles/条) 重叠，是 FlashAttention 高效的关键。

---

## 6. Causal Mask 的迭代优化

### 6.1 三种迭代次数

```cpp
num_iterations   = ceil_div(min(chunk_size, causal_bound), CTA_TILE_KV);
mask_iteration   = ... / CTA_TILE_KV;
window_iteration = ... / CTA_TILE_KV;
```

- `num_iterations`: 实际要循环多少个 KV tile。
- `mask_iteration`: causal mask 开始生效的 tile 索引。之前的 tile **完全不需要 mask** (所有元素都有效)。
- `window_iteration`: sliding window 开始 mask 的 tile 索引。

### 6.2 条件 mask

```cpp
if (iter >= mask_iteration || iter < window_iteration) {
    logits_mask(...);
}
```

对于 causal attention 的前几个 KV tile (全部落在因果范围内)，直接跳过 `logits_mask` 调用——省掉逐元素的条件判断。只有跨越因果边界的 tile 才需要精确 mask。

---

## 7. 完整数据流总览

```
gmem                    smem                    reg (per warp)           output
─────────────────────────────────────────────────────────────────────────────────
                                                        
Q ──cp.async──→ q_smem ──ldmatrix──→ a_frag ─┐
                                             │
K tile_i ─cp.async─→ k_smem ─ldmatrix──→ b_frag ─┤ mma.sync
                                             │     ↓
                                             ├──→ s_frag  (Q@K^T)
                                             │     │
                                             │     ↓ logits_transform + mask
                                             │     │
                                             │     ↓ update_mdo_states
                                             │     │
V tile_i ─cp.async─→ v_smem ─ldmatrix──→ b_frag' ┤   ↓
                                               │ mma.sync
                                               ↓
                                        o_frag  (P@V)  ──→ O (gmem)
                                               ↑
                                        d (分母累加)
                                        m (行最大值)
```

**Q 常驻 smem，K/V 流式过 smem，S/P 永远不离开寄存器。** 这就是 FlashAttention 用 `O(sqrt(N))` 的 smex 显存算 `O(N^2)` attention 的核心。

---

## 8. 与 Python API 的对应关系

| Python 参数 | Kernel 内部 |
|---|---|
| `q.shape = [qo_len, num_qo_heads, head_dim]` | `params.q`, `params.qo_len`, `params.num_qo_heads` |
| `causal=True` | `MASK_MODE = MaskMode::kCausal` |
| `pos_encoding_mode="ROPE_LLAMA"` | `POS_ENCODING_MODE = kRoPELlama` → 调 `q_smem_inplace_apply_rotary` |
| `backend="fa2"` | 走本文件 (非 Hopper 路径) |
| `backend="fa3"` | 走 `hopper/prefill_sm90.cuh` |
| `return_lse=True` | `params.lse != nullptr` → 循环后写 `lse` |

---

## 9. 进一步阅读

- **FlashAttention 原论文**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
- **在线 softmax 数学推导**: Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018)
- **PTX mma.sync**: PTX ISA §7.24, `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
- **ldmatrix 指令**: PTX ISA §7.25, `ldmatrix.sync.aligned.m8n8.x4`
- **cp.async 指令**: PTX ISA §7.6, `cp.async.ca.shared.global`
- **Hopper 路径 (FA3)**: `include/flashinfer/attention/hopper/prefill_sm90.cuh`, 用 TMA + WGMMA + warpgroup 替代手动 cp.async + mma.sync
