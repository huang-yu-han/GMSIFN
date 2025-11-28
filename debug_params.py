"""
Debug script to check parameter order in GMSIFN model
"""

import torch
from models import GMSIFN

# Create model
model = GMSIFN(
    radius=2,
    T=2,
    input_feature_dim=240,
    input_bond_dim=1,
    fingerprint_dim=240,
    output_units_num=16,
    p_dropout=0.2,
    top_k=5
)

# Get all parameters
params = list(model.parameters())

print("=" * 80)
print("GMSIFN Model Parameters")
print("=" * 80)
print(f"Total number of parameters: {len(params)}\n")

# Print all parameter shapes
for i, param in enumerate(params):
    print(f"params[{i:2d}]: shape {str(param.shape):20s} | size: {param.numel():7d}")

print("\n" + "=" * 80)
print("Expected Parameter Mapping (from code comments)")
print("=" * 80)
print("""
0-1:   node_fc
2-3:   neighbor_fc
4-7:   GRUCell[0]
8-11:  GRUCell[1]
12-13: align[0]
14-15: align[1]
16-17: attend[0]
18-19: attend[1]
20-23: graph_GRUCell
24-25: graph_align
26-27: graph_attend
28-29: output
30-31: graph_align2
32-33: graph_attend2
34-37: graph_GRUCell2
38-39: node_input
40-42: atrr (GGL)
""")

# Check if indices match
print("=" * 80)
print("Verification")
print("=" * 80)

try:
    # Check critical parameters
    print(f"\nGGL parameters (should be at indices 40-42):")
    print(f"  Expected: edge_weight [1], node_transform.weight [10, 240], node_transform.bias [10]")

    # Try to find GGL parameters
    ggl_params = []
    for i, param in enumerate(params):
        if param.shape == torch.Size([1]):
            ggl_params.append((i, 'edge_weight', param.shape))
        elif param.shape == torch.Size([10, 240]):
            ggl_params.append((i, 'node_transform.weight', param.shape))
        elif param.shape == torch.Size([10]):
            # Could be bias
            ggl_params.append((i, 'possible node_transform.bias', param.shape))

    print(f"\n  Found GGL-like parameters:")
    for idx, name, shape in ggl_params:
        print(f"    params[{idx}]: {name} {shape}")

except Exception as e:
    print(f"Error during verification: {e}")
