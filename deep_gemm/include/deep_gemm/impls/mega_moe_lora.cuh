#pragma once

#include <cuda_bf16.h>
#include <cutlass/cutlass.h>

namespace deep_gemm::mega_moe_lora {

// LoRA hooks for `sm100_fp8_fp4_mega_moe_impl` (template `kLoRARank > 0`; with 0 none of this is instantiated).
//
// Routed experts use the shared-outer layout: gate_up A is shared by all experts (applied outside the kernel,
// `xa = x A^T` per source token, BF16), gate_up B is per expert. Each source token's `xa` row sits in the
// symmetric buffer at `lora_xa_offset`; dispatch copies it next to the pulled token, and the L1 epilogue adds
//   d = sum_{r = 0 .. R-1} xa[r] * B_e[n, r]      (FP32 FMA chain, r ascending)
// to the FP32 accumulator before its BF16 rounding. A zero `d` keeps the accumulator bits unchanged (including
// -0), so rows without an adapter (`xa == 0`) and zero adapters (`B == 0`) give the kernel's bits without LoRA.

// Dispatch: copy the routed `xa` row of a pulled token from its source rank into the local ring slot.
template <uint32_t kRank, uint32_t kRowBytes, typename sym_buffer_t>
CUTLASS_DEVICE void pull_xa(const sym_buffer_t& sym_buffer, const uint64_t& lora_xa_offset,
                            const uint32_t& src_rank_idx, const uint32_t& src_token_idx,
                            nv_bfloat16* lora_xa_ring, const uint32_t& ring_token_idx,
                            const uint32_t& lane_idx) {
    constexpr uint32_t kNumWords = kRank / 2;
    static_assert(kRank % 8 == 0 and kNumWords <= 32, "Invalid LoRA rank");
    const auto local_row = reinterpret_cast<const uint32_t*>(
        sym_buffer.template get_base_ptr<uint8_t*>() + lora_xa_offset + static_cast<uint64_t>(src_token_idx) * kRowBytes);
    const auto remote_row = sym_buffer.map(local_row, src_rank_idx);
    if (lane_idx < kNumWords)
        reinterpret_cast<uint32_t*>(lora_xa_ring + static_cast<uint64_t>(ring_token_idx) * kRank)[lane_idx] = remote_row[lane_idx];
}

// L1 epilogue: `values[q]` holds TMEM lane `row0 + 8 q` (weight row) for the thread's two tokens (`.x`, `.y`), as
// loaded by two `SM100_TMEM_LOAD_16dp256b1x`. `b_row0` points at B row `row0` (row stride `kRank`), `xa_0/1` at the
// two tokens' `xa` rows. Tokens past the block's valid rows are left untouched.
template <uint32_t kRank>
CUTLASS_DEVICE void add_delta(float2 (&values)[4],
                              const nv_bfloat16* xa_0, const nv_bfloat16* xa_1,
                              const bool& valid_0, const bool& valid_1,
                              const nv_bfloat16* b_row0) {
    static_assert(kRank % 8 == 0, "Invalid LoRA rank");
    float d[4][2] = {};
    #pragma unroll
    for (uint32_t c = 0; c < kRank / 8; ++ c) {
        const uint4 x0 = __ldg(reinterpret_cast<const uint4*>(xa_0) + c);
        const uint4 x1 = __ldg(reinterpret_cast<const uint4*>(xa_1) + c);
        const auto* x0h = reinterpret_cast<const __nv_bfloat162*>(&x0);
        const auto* x1h = reinterpret_cast<const __nv_bfloat162*>(&x1);
        #pragma unroll
        for (uint32_t q = 0; q < 4; ++ q) {
            const uint4 b = __ldg(reinterpret_cast<const uint4*>(b_row0 + 8 * q * kRank) + c);
            const auto* bh = reinterpret_cast<const __nv_bfloat162*>(&b);
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++ k) {
                const float2 bf = __bfloat1622float2(bh[k]);
                const float2 xf0 = __bfloat1622float2(x0h[k]);
                const float2 xf1 = __bfloat1622float2(x1h[k]);
                d[q][0] = fmaf(xf0.x, bf.x, d[q][0]);
                d[q][0] = fmaf(xf0.y, bf.y, d[q][0]);
                d[q][1] = fmaf(xf1.x, bf.x, d[q][1]);
                d[q][1] = fmaf(xf1.y, bf.y, d[q][1]);
            }
        }
    }
    #pragma unroll
    for (uint32_t q = 0; q < 4; ++ q) {
        if (valid_0 and d[q][0] != 0.0f)
            values[q].x += d[q][0];
        if (valid_1 and d[q][1] != 0.0f)
            values[q].y += d[q][1];
    }
}

} // namespace deep_gemm::mega_moe_lora
