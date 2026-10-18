"""Export original FP16 q_proj spectra on CPU for readable report plots (no training)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_file', type=Path, help='Local original model.safetensors')
    args = parser.parse_args()
    torch.set_num_threads(8)
    out = Path(__file__).resolve().parent / 'Assignment1_results'
    expected = pd.read_csv(out / 'q4_minimum_ranks.csv').set_index('layer')
    spectra = {}
    with safe_open(args.model_file, framework='pt', device='cpu') as weights:
        for layer in (0, 7, 15):
            # Match the notebook: weights loaded in FP16, SVD computed in FP32.
            weight = weights.get_tensor(f'model.layers.{layer}.self_attn.q_proj.weight').half().float()
            values = torch.linalg.svdvals(weight)
            energy = values.square().cumsum(0) / values.square().sum() * 100
            rank = int(torch.searchsorted(energy, 50)) + 1
            assert rank == expected.loc[layer, 'minimum_rank_for_50pct_energy']
            spectra[f'layer_{layer}'] = values.numpy()
            print(f'Layer {layer}: rank {rank} verified')
    np.savez(out / 'q4_spectra.npz', **spectra)


if __name__ == '__main__':
    main()
