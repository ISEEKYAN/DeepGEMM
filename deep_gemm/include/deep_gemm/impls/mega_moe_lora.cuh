#pragma once

#include <cuda_bf16.h>
#include <cutlass/cutlass.h>

namespace deep_gemm::mega_moe_lora {

// LoRA hooks for `sm100_fp8_fp4_mega_moe_impl` (template `kLoRARank > 0`; with 0 none of this is instantiated).
//
// Routed experts use the shared-outer layout: gate_up A is shared by all experts (applied outside the kernel,
// `xa = x A^T` per source token, BF16), gate_up B is per expert. Each source token's `xa` row sits in the
// symmetric buffer at `lora_xa_offset` (routed `xa`, then the shared expert's w1 and w3 `xa`); dispatch copies
// the routed part next to the pulled token, and the L1 epilogue adds
//   d = sum_{r = 0 .. R-1} xa[r] * B_e[n, r]      (FP32 FMA chain, r ascending)
// to the FP32 accumulator before its BF16 rounding. A zero `d` keeps the accumulator bits unchanged (including
// -0), so rows without an adapter (`xa == 0`) and zero adapters (`B == 0`) give the kernel's bits without LoRA.
// The shared expert's w1 / w3 adapters are separate: its gate rows read the w1 `xa` and B, its up rows the w3 ones.

// Dispatch: copy the routed `xa` row of a pulled token from its source rank into the local ring slot.
template <uint32_t kRank, uint64_t kRowBytes, typename sym_buffer_t>
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

// L1 epilogue, for all `kNumAtoms` 8-token atoms of a store block at once: atom `a`'s `values[a]` (8 FP32, as loaded
// by two `SM100_TMEM_LOAD_16dp256b1x`) hold TMEM lanes `row0 + 8 q` (weight rows, as float2 `q`) for the thread's two
// tokens (`.x`, `.y`) `token0 + 8 a` and `+ 1`. With the granularity-8 gate/up interleave, q = 0, 2 are gate rows and
// q = 1, 3 up rows; up rows read `xa` from `kUpOffset` on (0 for routed experts, the w1 rank for the shared expert).
// `b_row0` points at B row `row0` (row stride `kRank`), `xa` at token row 0 of the block (row stride `kXaStride`).
// Rank chunks are the outer loop, so each B chunk is loaded once per store block; every value's FMA chain still runs
// in ascending r. Tokens from `num_valid_tokens` on are left untouched.
template <uint32_t kRank, uint32_t kNumAtoms, uint32_t kXaStride, uint32_t kUpOffset = 0>
CUTLASS_DEVICE void add_delta(uint32_t (&values)[kNumAtoms][8],
                              const nv_bfloat16* xa, const uint32_t& token0, const uint32_t& num_valid_tokens,
                              const nv_bfloat16* b_row0) {
    static_assert(kRank % 8 == 0, "Invalid LoRA rank");
    constexpr uint32_t kNumXa = kUpOffset == 0 ? 1 : 2;
    float d[kNumAtoms][4][2] = {};
    #pragma unroll
    for (uint32_t c = 0; c < kRank / 8; ++ c) {
        float2 bf[4][4];
        #pragma unroll
        for (uint32_t q = 0; q < 4; ++ q) {
            const uint4 b = __ldg(reinterpret_cast<const uint4*>(b_row0 + 8 * q * kRank) + c);
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++ k)
                bf[q][k] = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&b)[k]);
        }
        #pragma unroll
        for (uint32_t a = 0; a < kNumAtoms; ++ a) {
            #pragma unroll
            for (uint32_t t = 0; t < 2; ++ t) {
                const auto row = xa + static_cast<uint64_t>(token0 + 8 * a + t) * kXaStride;
                float2 xf[kNumXa][4];
                #pragma unroll
                for (uint32_t h = 0; h < kNumXa; ++ h) {
                    const uint4 x = __ldg(reinterpret_cast<const uint4*>(row + h * kUpOffset) + c);
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++ k)
                        xf[h][k] = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&x)[k]);
                }
                #pragma unroll
                for (uint32_t q = 0; q < 4; ++ q) {
                    const auto& x = xf[kNumXa == 2 ? q % 2 : 0];
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++ k) {
                        d[a][q][t] = fmaf(x[k].x, bf[q][k].x, d[a][q][t]);
                        d[a][q][t] = fmaf(x[k].y, bf[q][k].y, d[a][q][t]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t a = 0; a < kNumAtoms; ++ a) {
        auto v = reinterpret_cast<float2*>(values[a]);
        #pragma unroll
        for (uint32_t q = 0; q < 4; ++ q) {
            if (token0 + 8 * a < num_valid_tokens and d[a][q][0] != 0.0f)
                v[q].x += d[a][q][0];
            if (token0 + 8 * a + 1 < num_valid_tokens and d[a][q][1] != 0.0f)
                v[q].y += d[a][q][1];
        }
    }
}

// L1 epilogue, down LoRA input: store the BF16 activation (SwiGLU x top-k weight, before the FP8 cast) of the
// thread's two tokens and two intermediate columns (`act[k]` = column `col0 + 8 k`), rows `row_0` / `row_1` of `out`.
template <uint32_t kRowElems>
CUTLASS_DEVICE void export_act(nv_bfloat16* out, const float2 (&act)[2], const uint32_t& col0,
                               const uint64_t& row_0, const bool& valid_0, const bool& valid_1) {
    #pragma unroll
    for (uint32_t k = 0; k < 2; ++ k) {
        if (valid_0)
            out[row_0 * kRowElems + col0 + 8 * k] = __float2bfloat16_rn(act[k].x);
        if (valid_1)
            out[(row_0 + 1) * kRowElems + col0 + 8 * k] = __float2bfloat16_rn(act[k].y);
    }
}

} // namespace deep_gemm::mega_moe_lora
