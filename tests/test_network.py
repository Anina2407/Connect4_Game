import torch
from agents.alphazero.network import Connect4Net

def test_network_forward_pass():
    """Test that the network processes input and produces outputs of correct shapes."""
    net = Connect4Net()
    input_tensor = torch.randn(1, 3, 6, 7)  # Batch size 1, 3 channels, 6x7 board
    policy, value = net(input_tensor)
    assert policy.shape == (1, 7), f"Expected policy shape (1, 7), got {policy.shape}"
    assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"

def test_network_batch_processing():
    """Test that the network handles batch inputs correctly."""
    net = Connect4Net()
    input_tensor = torch.randn(8, 3, 6, 7)  # Batch size 8
    policy, value = net(input_tensor)
    assert policy.shape == (8, 7), f"Expected policy shape (8, 7), got {policy.shape}"
    assert value.shape == (8, 1), f"Expected value shape (8, 1), got {value.shape}"

def test_network_no_nan_or_inf():
    """Test that the network outputs do not contain NaN or Inf values."""
    net = Connect4Net()
    input_tensor = torch.randn(4, 3, 6, 7)  # Batch size 4
    policy, value = net(input_tensor)
    assert torch.isfinite(policy).all(), "Policy output contains NaN or Inf values"
    assert torch.isfinite(value).all(), "Value output contains NaN or Inf values"

def test_network_gradients():
    """Test that the network computes gradients correctly."""
    net = Connect4Net()
    input_tensor = torch.randn(2, 3, 6, 7, requires_grad=True)  # Batch size 2
    policy, value = net(input_tensor)
    loss = policy.sum() + value.sum()
    loss.backward()
    assert input_tensor.grad is not None, "Gradients were not computed for input tensor"