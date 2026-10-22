import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


_SUPPORTED_ACTIVATIONS = ('swiglu', 'swiglustep', 'situ')


def _validate_activation(activation: str) -> None:
    assert activation in _SUPPORTED_ACTIVATIONS, \
        f'Activation must be one of {_SUPPORTED_ACTIVATIONS}, got `{activation}`'


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None,
                 lora_rank: int = 0,
                 shared_lora_rank: int = 0):
        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

        # Init
        _validate_activation(activation)
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        # LoRA: per-token `xa` rows after the kernel's layout, so dispatch can pull a token's `xa` from its source
        # rank like its hidden row. Row = [routed (lora_rank) | shared w1 (shared_lora_rank) | shared w3], BF16
        assert shared_lora_rank == 0 or (lora_rank > 0 and num_shared_experts > 0)
        self.lora_rank, self.shared_lora_rank = lora_rank, shared_lora_rank
        self.lora_xa_offset = align(num_bytes, 128)
        lora_row = lora_rank + 2 * shared_lora_rank
        # Then `lora_ret` [num_topk, num_max_tokens_per_rank, lora_rank] BF16: where the down LoRA's per-slot `A_e a`
        # rows are written back to their token's rank (by the caller, after the kernel)
        self.lora_ret_offset = align(self.lora_xa_offset + num_max_tokens_per_rank * lora_row * 2, 128)
        if lora_rank > 0:
            num_bytes = self.lora_ret_offset + num_topk * num_max_tokens_per_rank * lora_rank * 2
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)
        self.lora_xa = self.lora_xa_ring = self.lora_ret = self.lora_peer_bases = None
        if lora_rank > 0:
            self.lora_xa = self.buffer[self.lora_xa_offset:self.lora_xa_offset + num_max_tokens_per_rank * lora_row * 2] \
                .view(torch.bfloat16).view(num_max_tokens_per_rank, lora_row)
            self.lora_ret = self.buffer[self.lora_ret_offset:num_bytes].view(torch.bfloat16).view(
                num_topk, num_max_tokens_per_rank, lora_rank)
            # Every rank's symmetric buffer base, for writes into a peer's `lora_ret`
            self.lora_peer_bases = torch.tensor(list(self.handle.buffer_ptrs), dtype=torch.int64, device='cuda')
            self.lora_xa_ring = torch.empty((self.l1_acts.shape[0], lora_rank), dtype=torch.bfloat16, device='cuda')
            # Down LoRA inputs exported by the kernel: BF16 activations of the routed pool rows (worst-case pool
            # size) and of the shared expert's token rows, each pool row's (rank, token, topk), and per local
            # expert its pool rows [start, start + count)
            num_experts_per_rank = num_experts // group.size()
            num_max_pool_tokens = _get_num_max_pool_tokens(
                group.size(), num_max_tokens_per_rank, num_topk, num_experts_per_rank)
            self.lora_act = torch.empty((num_max_pool_tokens, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
            self.lora_pool_meta = torch.empty((num_max_pool_tokens, 3), dtype=torch.int, device='cuda')
            self.lora_expert_info = torch.empty((num_experts_per_rank, 2), dtype=torch.int, device='cuda')
            self.lora_shared_act = torch.empty(
                (num_max_tokens_per_rank, intermediate_hidden * num_shared_experts),
                dtype=torch.bfloat16, device='cuda') if shared_lora_rank > 0 else None

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.lora_xa = None
        self.lora_xa_ring = None
        self.lora_ret = self.lora_peer_bases = None
        self.lora_act = self.lora_shared_act = self.lora_pool_meta = self.lora_expert_info = None


# TODO: remove this function
def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu',
                                 lora_rank: int = 0,
                                 shared_lora_rank: int = 0) -> SymmBuffer:
    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation, lora_rank=lora_rank, shared_lora_rank=shared_lora_rank
    )


def _get_num_max_pool_tokens(num_ranks: int, num_max_tokens_per_rank: int, num_topk: int,
                             num_experts_per_rank: int) -> int:
    # As `layout::get_num_max_pool_tokens` (max candidate BLOCK_M 240, their LCM 1920)
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    return align(num_max_recv_tokens * min(num_topk, num_experts_per_rank) + num_experts_per_rank * 239, 1920)


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    _validate_activation(activation)
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def transform_lora_b_for_mega_moe(lora_b_gate_up: torch.Tensor) -> torch.Tensor:
    """Per-expert gate_up LoRA B ([experts, 2 * intermediate, rank], gate rows then up rows) in the L1 weights'
    gate/up interleaved row order, as `fp8_fp4_mega_moe(lora_b_gate_up=...)` expects."""
    assert lora_b_gate_up.dim() == 3 and lora_b_gate_up.dtype == torch.bfloat16
    return _interleave_weights(lora_b_gate_up.contiguous())


def transform_shared_lora_b_for_mega_moe(lora_b_w1: torch.Tensor, lora_b_w3: torch.Tensor) -> torch.Tensor:
    """The shared expert's gate (w1) and up (w3) LoRA B ([shared intermediate, rank] each) as one
    [2 * shared intermediate, rank] tensor in the shared L1 weights' gate/up interleaved row order."""
    assert lora_b_w1.shape == lora_b_w3.shape and lora_b_w1.dtype == torch.bfloat16
    return _interleave_weights(torch.cat([lora_b_w1, lora_b_w3]).unsqueeze(0)).squeeze(0).contiguous()


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     activation_alpha: float = 1.0,
                     activation_beta: float = 0.0,
                     lora_b_gate_up: Optional[torch.Tensor] = None,
                     lora_b_shared_gate_up: Optional[torch.Tensor] = None):
    """`lora_b_gate_up` (from `transform_lora_b_for_mega_moe`) enables the routed gate_up LoRA hook: each token's
    `sym_buffer.lora_xa` row (`x A^T`, zero for tokens without an adapter) times B is added to the L1 accumulator
    before its BF16 rounding. A zero delta leaves the output bits of the kernel without LoRA.
    `lora_b_shared_gate_up` (from `transform_shared_lora_b_for_mega_moe`) does the same for the shared expert, whose
    gate / up rows read the w1 / w3 parts of the `lora_xa` row."""
    _validate_activation(activation)
    lora_rank = 0 if lora_b_gate_up is None else sym_buffer.lora_rank
    assert lora_b_gate_up is None or lora_rank > 0, 'The symmetric buffer was created without `lora_rank`'
    # The `lora_xa` row layout includes the shared part whenever the buffer has one, so both hooks go together
    assert lora_b_shared_gate_up is None or lora_rank > 0, 'The shared hook needs the routed one'
    assert lora_rank == 0 or (lora_b_shared_gate_up is not None) == (sym_buffer.shared_lora_rank > 0), \
        'Pass `lora_b_shared_gate_up` exactly when the buffer has `shared_lora_rank`'
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        activation_alpha, activation_beta,
        sym_buffer.lora_xa_ring if lora_rank > 0 else None,
        lora_b_gate_up, sym_buffer.lora_xa_offset if lora_rank > 0 else 0, lora_rank,
        lora_b_shared_gate_up, sym_buffer.shared_lora_rank if lora_b_shared_gate_up is not None else 0,
        *((sym_buffer.lora_act, sym_buffer.lora_shared_act, sym_buffer.lora_pool_meta, sym_buffer.lora_expert_info)
          if lora_rank > 0 else (None, None, None, None))
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True,
                  activation_alpha: float = 1.0,
                  activation_beta: float = 0.0):
    assert activation in ('swiglu', 'swiglustep'), \
        f'BF16 Mega MoE supports `swiglu` or `swiglustep`, got `{activation}`'
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        activation_alpha, activation_beta
    )
