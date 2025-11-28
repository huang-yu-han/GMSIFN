"""
Test script to verify GMSIFN model functionality
"""

import torch
import sys
sys.path.insert(0, '.')

from models import GMSIFN

def test_model():
    """Test if the model can run without errors"""

    # Model configuration
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

    # Move to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy input [batch_size=2, num_nodes=9, feature_dim=5120]
    batch_size = 2
    num_nodes = 9
    feature_dim = 5120

    dummy_input = torch.randn(batch_size, num_nodes, feature_dim).to(device)

    # Get model parameters (same way as MetaLearner)
    params = list(model.parameters())

    print("=" * 60)
    print("Testing GMSIFN Model")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of parameters: {len(params)}")

    # Debug: Print first few parameter shapes
    print(f"\nParameter shapes (first 5):")
    for i in range(min(5, len(params))):
        print(f"  params[{i}]: {params[i].shape}")

    # Test forward pass
    try:
        print("\n[Test 1] Forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(dummy_input, params)
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape} (expected: [{batch_size}, 16])")

        if output.shape == (batch_size, 16):
            print(f"  ✓ Output shape is correct!")
        else:
            print(f"  ✗ Output shape mismatch!")
            return False

    except Exception as e:
        print(f"✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        return False

    # Test get_features
    try:
        print("\n[Test 2] Feature extraction...")
        with torch.no_grad():
            features = model.get_features(dummy_input, params)
        print(f"✓ Feature extraction successful!")
        print(f"  Feature shape: {features.shape} (expected: [{batch_size}, 240])")

        if features.shape == (batch_size, 240):
            print(f"  ✓ Feature shape is correct!")
        else:
            print(f"  ✗ Feature shape mismatch!")
            return False

    except Exception as e:
        print(f"✗ Feature extraction failed with error:")
        print(f"  {type(e).__name__}: {e}")
        return False

    # Test gradient computation
    try:
        print("\n[Test 3] Backward pass (gradient computation)...")
        model.train()
        output = model(dummy_input, params)
        dummy_target = torch.randint(0, 16, (batch_size,)).to(device)
        loss = torch.nn.functional.cross_entropy(output, dummy_target)
        loss.backward()
        print(f"✓ Backward pass successful!")
        print(f"  Loss value: {loss.item():.4f}")

    except Exception as e:
        print(f"✗ Backward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed! Model is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
