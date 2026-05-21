#!/usr/bin/env python3
"""Select sparse parent edges from saved YAQA cross-Hessian factors.

The script scans `{label}_cross{partner_gidx}_{hin,hout}.pt` files, computes a
relative cross-block strength

    rho = rho_I * rho_O
        = ||H_I(j,k)||_F / sqrt(||H_I(j,j)||_F ||H_I(k,k)||_F)
        * ||H_O(j,k)||_F / sqrt(||H_O(j,j)||_F ||H_O(k,k)||_F)

with the same dimension normalization used by visualize_cross_coupling.py, then
writes a parent-map JSON that quantize_cross_hess_llama.py can consume.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import torch

LAYER_ORDER = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']


def infer_n(flat_len: int) -> int:
    return int(round((-1 + (1 + 8 * flat_len) ** 0.5) / 2))


def flat_to_sym(v: torch.Tensor, n: int) -> torch.Tensor:
    A = torch.zeros(n, n, dtype=v.dtype)
    idx = torch.tril_indices(n, n)
    A[idx[0], idx[1]] = v
    A[idx[1], idx[0]] = v
    return A


def parse_label(label: str):
    layer_s, name = label.split('_', 1)
    return int(layer_s), name


def discover_entries(hess_path: str, names: list[str]):
    entries = {}
    valid_names = set(names)
    for fp in sorted(glob.glob(os.path.join(hess_path, '*_hin.pt'))):
        stem = os.path.basename(fp)[:-len('_hin.pt')]
        if '_cross' in stem:
            continue
        try:
            layer, name = parse_label(stem)
        except ValueError:
            continue
        if name not in valid_names:
            continue
        gidx = layer * len(LAYER_ORDER) + LAYER_ORDER.index(name)
        hout_fp = os.path.join(hess_path, f'{stem}_hout.pt')
        if not os.path.exists(hout_fp):
            continue
        entries[gidx] = {
            'label': stem,
            'layer': layer,
            'name': name,
            'gidx': gidx,
            'hin_path': fp,
            'hout_path': hout_fp,
        }
    return entries


def discover_cross_gidxs(hess_path: str, entries: dict[int, dict]) -> set[int]:
    label_to_gidx = {e['label']: g for g, e in entries.items()}
    used = set()
    for e in entries.values():
        label = e['label']
        for fp in glob.glob(os.path.join(hess_path, f'{label}_cross*_hin.pt')):
            base = os.path.basename(fp)
            mid = base[len(label) + len('_cross'):]
            try:
                partner_gidx = int(mid[:mid.index('_')])
            except ValueError:
                continue
            if partner_gidx in entries:
                used.add(e['gidx'])
                used.add(partner_gidx)
    return used


def load_diagonal_stats(entries: dict[int, dict], dtype):
    for e in entries.values():
        hin_flat = torch.load(e['hin_path'], map_location='cpu').to(dtype)
        hout_flat = torch.load(e['hout_path'], map_location='cpu').to(dtype)
        n = infer_n(len(hin_flat))
        m = infer_n(len(hout_flat))
        hin = flat_to_sym(hin_flat, n)
        hout = flat_to_sym(hout_flat, m)
        e['n'] = n
        e['m'] = m
        e['hin_norm_scaled'] = (hin.norm() / max(m, 1)).item()
        e['hout_norm_scaled'] = (hout.norm() / max(n, 1)).item()


def relative_strength(cross, left_norm, right_norm):
    denom = (left_norm * right_norm) ** 0.5
    if denom <= 0:
        return float('nan')
    return (cross.norm().item() / denom)


def collect_records(hess_path: str, entries: dict[int, dict], dtype):
    label_to_gidx = {e['label']: g for g, e in entries.items()}
    records = []
    seen = set()
    for owner_gidx, owner in sorted(entries.items()):
        owner_label = owner['label']
        pattern = os.path.join(hess_path, f'{owner_label}_cross*_hin.pt')
        for hin_fp in sorted(glob.glob(pattern)):
            base = os.path.basename(hin_fp)
            mid = base[len(owner_label) + len('_cross'):]
            try:
                partner_gidx = int(mid[:mid.index('_')])
            except ValueError:
                continue
            if partner_gidx not in entries:
                continue
            hout_fp = hin_fp[:-len('_hin.pt')] + '_hout.pt'
            if not os.path.exists(hout_fp):
                continue
            pair_key = tuple(sorted((owner_gidx, partner_gidx)))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            partner = entries[partner_gidx]
            C_in = torch.load(hin_fp, map_location='cpu').to(dtype)
            C_out = torch.load(hout_fp, map_location='cpu').to(dtype)

            expected_in = (partner['n'], owner['n'])
            expected_out = (partner['m'], owner['m'])
            if C_in.shape != expected_in or C_out.shape != expected_out:
                print(f'[warn] shape mismatch for {owner_label}<->{partner["label"]}, skipping')
                continue

            rho_i = relative_strength(C_in, owner['hin_norm_scaled'], partner['hin_norm_scaled'])
            rho_o = relative_strength(C_out, owner['hout_norm_scaled'], partner['hout_norm_scaled'])
            rho = rho_i * rho_o

            if owner_gidx < partner_gidx:
                parent, child = owner, partner
            else:
                parent, child = partner, owner

            layer_gap = child['layer'] - parent['layer']
            gidx_gap = child['gidx'] - parent['gidx']
            records.append({
                'parent': parent['label'],
                'child': child['label'],
                'parent_gidx': parent['gidx'],
                'child_gidx': child['gidx'],
                'parent_name': parent['name'],
                'child_name': child['name'],
                'pair_type': f'{parent["name"]}->{child["name"]}',
                'parent_layer': parent['layer'],
                'child_layer': child['layer'],
                'layer_gap': layer_gap,
                'gidx_gap': gidx_gap,
                'rho_i': rho_i,
                'rho_o': rho_o,
                'rho': rho,
            })
    return records


def select_pair_types(records, top_pair_types: int, min_type_strength: float,
                      min_layer_support: int, same_block_only: bool):
    groups = defaultdict(list)
    for r in records:
        if same_block_only and r['layer_gap'] != 0:
            continue
        groups[r['pair_type']].append(r)

    summaries = []
    for pair_type, items in groups.items():
        layers = {r['child_layer'] for r in items}
        mean = sum(r['rho'] for r in items) / len(items)
        max_v = max(r['rho'] for r in items)
        summaries.append({
            'pair_type': pair_type,
            'count': len(items),
            'layer_support': len(layers),
            'mean_rho': mean,
            'max_rho': max_v,
        })
    summaries.sort(key=lambda x: (-x['mean_rho'], -x['max_rho'], x['pair_type']))

    selected = []
    for s in summaries:
        if s['layer_support'] < min_layer_support:
            continue
        if min_type_strength > 0 and s['mean_rho'] < min_type_strength:
            continue
        selected.append(s)
    if top_pair_types > 0:
        selected = selected[:top_pair_types]
    return summaries, selected


def select_edges(records, args):
    candidates = list(records)
    if args.same_block_only:
        candidates = [r for r in candidates if r['layer_gap'] == 0]
    if args.max_gidx_gap >= 0:
        candidates = [r for r in candidates if r['gidx_gap'] <= args.max_gidx_gap]

    type_summaries, selected_types = select_pair_types(
        records,
        args.top_pair_types,
        args.min_pair_type_strength,
        args.min_layer_support,
        args.same_block_only,
    )
    selected_type_set = {s['pair_type'] for s in selected_types}

    selected = []
    if selected_type_set:
        selected.extend(r for r in candidates if r['pair_type'] in selected_type_set)

    if args.threshold > 0:
        selected.extend(r for r in candidates if r['rho'] >= args.threshold)

    if args.top_edges > 0:
        selected.extend(sorted(candidates, key=lambda r: -r['rho'])[:args.top_edges])

    if args.top_per_child > 0:
        by_child = defaultdict(list)
        for r in candidates:
            by_child[r['child']].append(r)
        for child_items in by_child.values():
            selected.extend(sorted(child_items, key=lambda r: -r['rho'])[:args.top_per_child])

    if not (selected_type_set or args.threshold > 0 or args.top_edges > 0 or args.top_per_child > 0):
        selected = []

    unique = {}
    for r in selected:
        unique[(r['parent_gidx'], r['child_gidx'])] = r
    return sorted(unique.values(), key=lambda r: (r['child_gidx'], r['parent_gidx'])), type_summaries, selected_types


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hess_path', required=True)
    parser.add_argument('--output_path', default=None,
                        help='JSON parent-map output. Default: <hess_path>/selected_parent_edges.json')
    parser.add_argument('--summary_csv', default=None,
                        help='Per-edge CSV. Default: <hess_path>/cross_pair_strength.csv')
    parser.add_argument('--type_csv', default=None,
                        help='Per-pair-type CSV. Default: <hess_path>/cross_pair_type_strength.csv')
    parser.add_argument('--names', default=','.join(LAYER_ORDER))
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Select every directed edge with rho >= threshold.')
    parser.add_argument('--top_edges', type=int, default=0,
                        help='Select globally top-K directed edges by rho.')
    parser.add_argument('--top_per_child', type=int, default=0,
                        help='Select top-K parents for each child weight.')
    parser.add_argument('--top_pair_types', type=int, default=0,
                        help='Select all edges belonging to the top-K parent->child name types by mean rho.')
    parser.add_argument('--min_pair_type_strength', type=float, default=0.0,
                        help='Select all pair types whose mean rho is at least this value.')
    parser.add_argument('--min_layer_support', type=int, default=1,
                        help='Pair type must appear in at least this many child layers.')
    parser.add_argument('--same_block_only', action='store_true',
                        help='Only rank/select pairs within the same transformer block.')
    parser.add_argument('--max_gidx_gap', type=int, default=-1,
                        help='Only select edges with child_gidx - parent_gidx <= this value. -1 disables.')
    parser.add_argument('--print_top', type=int, default=30)
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    names = [x.strip() for x in args.names.split(',') if x.strip()]
    output_path = args.output_path or os.path.join(args.hess_path, 'selected_parent_edges.json')
    summary_csv = args.summary_csv or os.path.join(args.hess_path, 'cross_pair_strength.csv')
    type_csv = args.type_csv or os.path.join(args.hess_path, 'cross_pair_type_strength.csv')

    entries = discover_entries(args.hess_path, names)
    if not entries:
        raise SystemExit(f'No diagonal Hessian files found in {args.hess_path!r}')

    cross_gidxs = discover_cross_gidxs(args.hess_path, entries)
    if cross_gidxs:
        entries = {g: e for g, e in entries.items() if g in cross_gidxs}
        load_diagonal_stats(entries, dtype)
        records = collect_records(args.hess_path, entries, dtype)
        records.sort(key=lambda r: -r['rho'])
    else:
        records = []

    selected, type_summaries, selected_types = select_edges(records, args)
    parent_map = defaultdict(list)
    for r in selected:
        parent_map[r['child']].append(r['parent'])
    parent_map = {k: sorted(v, key=lambda label: int(label.split('_', 1)[0]) * 7 + LAYER_ORDER.index(label.split('_', 1)[1]))
                  for k, v in sorted(parent_map.items())}

    edge_fields = ['parent', 'child', 'parent_gidx', 'child_gidx',
                   'parent_name', 'child_name', 'pair_type',
                   'parent_layer', 'child_layer', 'layer_gap', 'gidx_gap',
                   'rho_i', 'rho_o', 'rho']
    write_csv(summary_csv, records, edge_fields)
    type_fields = ['pair_type', 'count', 'layer_support', 'mean_rho', 'max_rho']
    write_csv(type_csv, type_summaries, type_fields)

    payload = {
        'format': 'yaqa_cross_parent_edges_v1',
        'hess_path': os.path.abspath(args.hess_path),
        'metric': 'rho_i * rho_o',
        'selection_args': vars(args),
        'selected_pair_types': selected_types,
        'edges': selected,
        'parent_map': parent_map,
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Found {len(records)} cross edges')
    print(f'Selected {len(selected)} parent edges')
    print(f'Wrote {output_path}')
    print(f'Wrote {summary_csv}')
    print(f'Wrote {type_csv}')
    if args.print_top > 0:
        print('\nTop edges:')
        for r in records[:args.print_top]:
            mark = '*' if any(r['parent'] == s['parent'] and r['child'] == s['child'] for s in selected) else ' '
            print(f'{mark} {r["parent"]:>10} -> {r["child"]:<10} '
                  f'{r["pair_type"]:>10} rho={r["rho"]:.6g} '
                  f'rho_i={r["rho_i"]:.4g} rho_o={r["rho_o"]:.4g}')
        print('\nTop pair types:')
        for s in type_summaries[:args.print_top]:
            mark = '*' if s in selected_types else ' '
            print(f'{mark} {s["pair_type"]:>10} mean={s["mean_rho"]:.6g} '
                  f'max={s["max_rho"]:.6g} support={s["layer_support"]} count={s["count"]}')


if __name__ == '__main__':
    main()
