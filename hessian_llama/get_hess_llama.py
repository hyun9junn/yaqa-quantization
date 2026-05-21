import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from functools import partial

import torch
import torch.distributed as dist
from custom_linear_A import CustomLinear as CLA
from custom_linear_B import CustomLinear as CLB
from data_utils import DataLoader, FullCtx
from datasets import load_dataset
from llama_hess import LlamaForCausalLM
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
from torch.distributed.fsdp.wrap import (enable_wrap,
                                         transformer_auto_wrap_policy, wrap)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

import argparse

import tqdm
from accelerate import init_empty_weights
from contextlib import contextmanager

@contextmanager
def isolate_rng():
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--n_seqs', default=65536, type=int)
parser.add_argument('--ctx_size', default=2048, type=int)
parser.add_argument('--power_iters', default=1, type=int)
parser.add_argument('--start_layer', default=0, type=int)
parser.add_argument('--end_layer', default=100000, type=int)
parser.add_argument('--hessian_sketch', default='B', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--orig_model', type=str)
parser.add_argument('--cpu_offload', action='store_true')
parser.add_argument('--fp64_accum', action='store_true')
parser.add_argument('--cross', action='store_true', default=False)
parser.add_argument('--parent_band',
                    default=1,
                    type=int,
                    help='Collect cross-Hessians for all weight pairs within this gidx distance. '
                         'Ignored when --parent_block_window is set.')
parser.add_argument('--parent_block_window',
                    default=-1,
                    type=int,
                    help='Alternative to --parent_band: collect cross-Hessians for all weight '
                         'pairs whose transformer block indices differ by at most this value. '
                         'Window=1 collects all pairs across adjacent blocks (7×7=49 pairs per '
                         'boundary). Set to -1 (default) to use --parent_band instead.')
parser.add_argument('--parent_extra_pairs',
                    default='',
                    type=str,
                    help='Always collect these additional cross-Hessian pairs regardless of band. '
                         'Format: "labelA,labelB;..." where labels are block_name (e.g. "0_q"). '
                         'Use "*" as a block-index wildcard to expand across all layers: '
                         '"*_q,*_v" collects q↔v for every block. '
                         'Names: q, k, v, o, up, gate, down.')
parser.add_argument('--local_als_iters', default=3, type=int)
args = parser.parse_args()
if args.parent_band < 0:
    raise ValueError('--parent_band must be non-negative')
if args.parent_block_window == 0:
    raise ValueError('--parent_block_window must be >= 1 (or -1 to disable)')


LAYER_ORDER = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']


def label_to_gidx(label: str) -> int:
    """Convert weight label like '3_gate' to global weight index."""
    block_str, *name_parts = label.split('_')
    name = '_'.join(name_parts)
    return int(block_str) * len(LAYER_ORDER) + LAYER_ORDER.index(name)


def parse_extra_pairs(extra_pairs_str: str, num_layers: int) -> frozenset:
    """Parse pair spec into frozenset of (min_gidx, max_gidx).

    Use '*' as a wildcard for the block index to expand a pattern across all layers.
    Examples:
        "0_q,0_v"          one explicit pair
        "*_q,*_v"           q↔v for every block  (expands to 0_q,0_v;1_q,1_v;...)
        "0_q,0_v;*_gate,*_down"  mix of explicit and wildcard
    """
    pairs = set()
    if not extra_pairs_str.strip():
        return frozenset()
    for pair in extra_pairs_str.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        parts = [p.strip() for p in pair.split(',')]
        if len(parts) != 2:
            raise ValueError(f'Invalid pair (expected "labelA,labelB"): {pair!r}')
        l1, l2 = parts
        if '*' in l1 or '*' in l2:
            for i in range(num_layers):
                g1 = label_to_gidx(l1.replace('*', str(i)))
                g2 = label_to_gidx(l2.replace('*', str(i)))
                pairs.add((min(g1, g2), max(g1, g2)))
        else:
            g1 = label_to_gidx(l1)
            g2 = label_to_gidx(l2)
            pairs.add((min(g1, g2), max(g1, g2)))
    return frozenset(pairs)


def setup(rank, world_size):
    from datetime import timedelta
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(days=1))


def cleanup():
    dist.destroy_process_group()


local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

setup(local_rank, local_world_size)

cutoff = args.n_seqs // (local_world_size * args.batch_size)
if local_rank == 0:
    print(f'USING {cutoff} SEQUENCES PER GPU')

model = LlamaForCausalLM.from_pretrained(args.orig_model,
                                         torch_dtype='auto',
                                         device_map='cpu')

extra_pairs = parse_extra_pairs(args.parent_extra_pairs, len(model.model.layers))
if local_rank == 0 and extra_pairs:
    print(f'Extra cross-Hessian pairs (gidx): {sorted(extra_pairs)}')

# Resolve filter mode: block_adjacent takes priority over gidx_band when set.
if args.parent_block_window >= 1:
    _cross_filter = 'block_adjacent'
    _cross_window = args.parent_block_window
    if local_rank == 0:
        print(f'Cross-Hessian filter: block_adjacent (window={_cross_window})')
else:
    _cross_filter = 'gidx_band'
    _cross_window = args.parent_band
    if local_rank == 0:
        print(f'Cross-Hessian filter: gidx_band (band={_cross_window})')

if args.hessian_sketch == 'A':
    custom_linear_layer = CLA
    if args.power_iters < 2:
        if local_rank == 0:
            print(
                'ERROR: Must use more than half a round of power iteration for A'
            )
        raise Exception
elif args.hessian_sketch == 'B':
    custom_linear_layer = CLB
else:
    raise Exception

device_ct = 0
with torch.autograd.set_grad_enabled(False):
    gidx = 0
    for i in range(len(model.model.layers)):
        l = model.model.layers[i]
        collect_hess = (i >= args.start_layer and i < args.end_layer)
        args.fp64_accum = False

        name = f'{i}_q'
        new_q = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    name,
                                    gidx, 'q',
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.q_proj.in_features,
                                    l.self_attn.q_proj.out_features,
                                    block_idx=i,
                                    cross_filter=_cross_filter,
                                    cross_block_window=_cross_window,
                                    extra_pairs=extra_pairs,
                                    dtype=l.self_attn.q_proj.weight.dtype)
        new_q.weight = l.self_attn.q_proj.weight
        del l.self_attn.q_proj
        l.self_attn.q_proj = new_q
        device_ct += 1
        gidx += 1

        name = f'{i}_k'
        new_k = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    name,
                                    gidx, 'k',
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.k_proj.in_features,
                                    l.self_attn.k_proj.out_features,
                                    block_idx=i,
                                    cross_filter=_cross_filter,
                                    cross_block_window=_cross_window,
                                    extra_pairs=extra_pairs,
                                    dtype=l.self_attn.k_proj.weight.dtype)
        new_k.weight = l.self_attn.k_proj.weight
        del l.self_attn.k_proj
        l.self_attn.k_proj = new_k
        device_ct += 1
        gidx += 1

        name = f'{i}_v'
        new_v = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    name,
                                    gidx, 'v',
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.v_proj.in_features,
                                    l.self_attn.v_proj.out_features,
                                    block_idx=i,
                                    cross_filter=_cross_filter,
                                    cross_block_window=_cross_window,
                                    extra_pairs=extra_pairs,
                                    dtype=l.self_attn.v_proj.weight.dtype)
        new_v.weight = l.self_attn.v_proj.weight
        del l.self_attn.v_proj
        l.self_attn.v_proj = new_v
        device_ct += 1
        gidx += 1

        name = f'{i}_o'
        new_o = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    name,
                                    gidx, 'o',
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.o_proj.in_features,
                                    l.self_attn.o_proj.out_features,
                                    block_idx=i,
                                    cross_filter=_cross_filter,
                                    cross_block_window=_cross_window,
                                    extra_pairs=extra_pairs,
                                    dtype=l.self_attn.o_proj.weight.dtype)
        new_o.weight = l.self_attn.o_proj.weight
        del l.self_attn.o_proj
        l.self_attn.o_proj = new_o
        device_ct += 1
        gidx += 1

        name = f'{i}_up'
        new_up = custom_linear_layer(device_ct % local_world_size,
                                     args.cpu_offload,
                                     name,
                                     gidx, 'up',
                                     collect_hess,
                                     args.fp64_accum,
                                     l.mlp.up_proj.in_features,
                                     l.mlp.up_proj.out_features,
                                     block_idx=i,
                                     cross_filter=_cross_filter,
                                     cross_block_window=_cross_window,
                                     extra_pairs=extra_pairs,
                                     dtype=l.mlp.up_proj.weight.dtype)
        new_up.weight = l.mlp.up_proj.weight
        del l.mlp.up_proj
        l.mlp.up_proj = new_up
        device_ct += 1
        gidx += 1

        name = f'{i}_gate'
        new_gate = custom_linear_layer(device_ct % local_world_size,
                                       args.cpu_offload,
                                       name,
                                       gidx, 'gate',
                                       collect_hess,
                                       args.fp64_accum,
                                       l.mlp.gate_proj.in_features,
                                       l.mlp.gate_proj.out_features,
                                       block_idx=i,
                                       cross_filter=_cross_filter,
                                       cross_block_window=_cross_window,
                                       extra_pairs=extra_pairs,
                                       dtype=l.mlp.gate_proj.weight.dtype)
        new_gate.weight = l.mlp.gate_proj.weight
        del l.mlp.gate_proj
        l.mlp.gate_proj = new_gate
        device_ct += 1
        gidx += 1

        name = f'{i}_down'
        new_down = custom_linear_layer(device_ct % local_world_size,
                                       args.cpu_offload,
                                       name,
                                       gidx, 'down',
                                       collect_hess,
                                       args.fp64_accum,
                                       l.mlp.down_proj.in_features,
                                       l.mlp.down_proj.out_features,
                                       block_idx=i,
                                       cross_filter=_cross_filter,
                                       cross_block_window=_cross_window,
                                       extra_pairs=extra_pairs,
                                       dtype=l.mlp.down_proj.weight.dtype)
        new_down.weight = l.mlp.down_proj.weight
        del l.mlp.down_proj
        l.mlp.down_proj = new_down
        device_ct += 1
        gidx += 1

auto_wrap_policy = partial(transformer_auto_wrap_policy,
                           transformer_layer_cls={
                               type(model.model.layers[0]),
                           })

model = FSDP(model,
             device_id=local_rank,
             auto_wrap_policy=auto_wrap_policy,
             cpu_offload=CPUOffload(offload_params=True),
             use_orig_params=False)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=(lambda module: not type(module) == custom_linear_layer),
)

torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained(args.orig_model)

batch = torch.zeros(args.batch_size,
                    args.ctx_size,
                    dtype=torch.int64,
                    device=local_rank)

for pit in range(args.power_iters):

    if local_rank == 0:
        print(f'POWER ITERATION {pit}')
        dataset = load_dataset('allenai/c4', 'en', split='train',
                               streaming=True).shuffle(seed=args.seed, buffer_size=10000)
        dl = iter(
            torch.utils.data.DataLoader(
                FullCtx(iter(dataset), tok, args.ctx_size),
                batch_size=args.batch_size * local_world_size,
                num_workers=1))

    range_counter = range(cutoff)
    if local_rank == 0:
        range_counter = tqdm.tqdm(range_counter)
    for i in range_counter:
        blist = list(
            torch.split(next(dl).to(local_rank), args.batch_size,
                        dim=0)) if local_rank == 0 else None

        torch.distributed.scatter(batch, blist, src=0)
        logits = model(batch,
                       mode=(pit, i == 0, i == (cutoff - 1), args.cross, args.local_als_iters),
                       use_cache=False)['logits']
        logits = logits.view(-1, logits.shape[-1]).float()

        with torch.no_grad():
            with isolate_rng():
                torch.manual_seed(i)
                fake_target = torch.distributions.categorical.Categorical(
                    logits=logits).sample()

        torch.nn.functional.cross_entropy(logits, fake_target).backward()

        if i == cutoff - 1:
            for l in model.modules():
                if hasattr(l, 'hin'):
                    ct = max(l.ct, 1)
                    torch.save(
                        l.hin / ct,
                        os.path.join(args.save_path, f'{l.fname}_hin.pt'))
                    torch.save(
                        l.hout / ct,
                        os.path.join(args.save_path, f'{l.fname}_hout.pt'))
                    
                if hasattr(l, 'cross_hin'):
                    for other_idx, tensor in l.cross_hin.items():
                        torch.save(
                            tensor / ct,
                            os.path.join(args.save_path,
                                         f'{l.fname}_cross{other_idx}_hin.pt'))
                if hasattr(l, 'cross_hout'):
                    for other_idx, tensor in l.cross_hout.items():
                        torch.save(
                            tensor / ct,
                            os.path.join(args.save_path,
                                         f'{l.fname}_cross{other_idx}_hout.pt'))

            print(f'RANK {local_rank} SAVED CURRENT HESSIANS')
