import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class BoardDataset(Dataset):
    """
    Dataset class for Connect 4 board states, policies, and values.
    """
    def __init__(self, data):  # data = np.array of (state, policy, value)
        self.states = data[:, 0]
        self.policies = data[:, 1]
        self.values = data[:, 2]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = np.int64(self.states[index].transpose(2, 0, 1))
        return state, self.policies[index], self.values[index]


class InitialConvLayer(nn.Module):
    """ 
    InitialConvLayer is a PyTorch module representing the initial convolutional layer for processing the input board state in a Connect4 environment.
    Args:
        input_channels (int): Number of input channels, typically corresponding to the board representation (default: 3).
        output_channels (int): Number of output channels for the convolutional layer (default: 128).
        kernel_size (int): Size of the convolutional kernel (default: 3).
    Attributes:
        conv (nn.Conv2d): 2D convolutional layer.
        batch_norm (nn.BatchNorm2d): Batch normalization layer applied after convolution.
    Forward Input:
        x (torch.Tensor): Input tensor representing the board state, expected to be reshaped to (-1, 3, 6, 7).
    Forward Output:
        torch.Tensor: Output tensor after applying convolution, batch normalization, and ReLU activation.
    """
    def __init__(self, input_channels=3, output_channels=128, kernel_size=3):
        """
        Initializes the convolutional neural network layer with specified parameters.

        Args:
            input_channels (int, optional): Number of input channels for the convolutional layer. Defaults to 3.
            output_channels (int, optional): Number of output channels for the convolutional layer. Defaults to 128.
            kernel_size (int or tuple, optional): Size of the convolutional kernel. Defaults to 3.

        Attributes:
            conv (nn.Conv2d): 2D convolutional layer.
            batch_norm (nn.BatchNorm2d): Batch normalization layer applied after convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor, expected to be reshaped to (-1, 3, 6, 7) to match the board dimensions.

        Returns:
            torch.Tensor: Output tensor after applying convolution, batch normalization, and ReLU activation.
        """
        x = x.view(-1, 3, 6, 7)  # Reshape to match board dimensions
        return F.relu(self.batch_norm(self.conv(x)))


class ResidualLayer(nn.Module):
    """
    A single residual layer for convolutional neural networks, consisting of two convolutional layers with batch normalization and a skip connection.
    Args:
        channels (int): Number of input and output channels for the convolutional layers. Default is 128.
    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
    Forward Output:
        torch.Tensor: Output tensor of the same shape as input, after applying two convolutional layers, batch normalization, ReLU activations, and a residual (skip) connection.
    """
    def __init__(self, channels=128):
        """
        Initializes the neural network block with two convolutional layers and corresponding batch normalization layers.

        Args:
            channels (int, optional): Number of input and output channels for the convolutional layers. Defaults to 128.

        Attributes:
            conv1 (nn.Conv2d): First convolutional layer with kernel size 3x3, no bias.
            batch_norm1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
            conv2 (nn.Conv2d): Second convolutional layer with kernel size 3x3, no bias.
            batch_norm2 (nn.BatchNorm2d): Batch normalization layer after the second convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Performs a forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying two convolutional layers with batch normalization and a residual connection, followed by a ReLU activation.
        """
        residual = x
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        return F.relu(x + residual)


class OutputLayer(nn.Module):
    """
    OutputLayer is a neural network module designed for AlphaZero-style agents, providing separate output heads for policy and value predictions.
    Args:
        board_dims (tuple, optional): Dimensions of the game board as (rows, columns). Default is (6, 7) for Connect4.
        policy_channels (int, optional): Number of channels in the policy head convolution. Default is 32.
        value_channels (int, optional): Number of channels in the value head convolution. Default is 3.
    Attributes:
        value_conv (nn.Conv2d): Convolutional layer for the value head.
        value_bn (nn.BatchNorm2d): Batch normalization for the value head.
        value_fc1 (nn.Linear): First fully connected layer for the value head.
        value_fc2 (nn.Linear): Second fully connected layer for the value head, outputs a scalar value.
        policy_conv (nn.Conv2d): Convolutional layer for the policy head.
        policy_bn (nn.BatchNorm2d): Batch normalization for the policy head.
        policy_fc (nn.Linear): Fully connected layer for the policy head, outputs logits for each possible action.
    Methods:
        forward(x):
            Computes the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 128, board_rows, board_cols).
            Returns:
                p (torch.Tensor): Policy probabilities for each action (batch_size, board_cols).
                v (torch.Tensor): Value prediction for each input (batch_size, 1).
    """
    def __init__(self, board_dims=(6, 7), policy_channels=32, value_channels=3):
        """
        Initializes the neural network layers for the AlphaZero agent's value and policy heads.
        Args:
            board_dims (tuple, optional): Dimensions of the game board as (rows, columns). Defaults to (6, 7).
            policy_channels (int, optional): Number of output channels for the policy convolutional layer. Defaults to 32.
            value_channels (int, optional): Number of output channels for the value convolutional layer. Defaults to 3.
        Attributes:
            value_conv (nn.Conv2d): Convolutional layer for the value head.
            value_bn (nn.BatchNorm2d): Batch normalization for the value head.
            value_fc1 (nn.Linear): First fully connected layer for the value head.
            value_fc2 (nn.Linear): Second fully connected layer for the value head.
            policy_conv (nn.Conv2d): Convolutional layer for the policy head.
            policy_bn (nn.BatchNorm2d): Batch normalization for the policy head.
            policy_fc (nn.Linear): Fully connected layer for the policy head.
        """
        super().__init__()
        self.value_conv = nn.Conv2d(128, value_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_dims[0] * board_dims[1], 32)
        self.value_fc2 = nn.Linear(32, 1)

        self.policy_conv = nn.Conv2d(128, policy_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * board_dims[0] * board_dims[1], board_dims[1])

    def forward(self, x):
        """
        Performs a forward pass through the neural network, producing policy and value outputs.
        Args:
            x (torch.Tensor): Input tensor representing the game state, typically of shape (batch_size, channels, height, width).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - p: Policy tensor with probabilities for each possible action, shape (batch_size, num_actions).
                - v: Value tensor with predicted value for each input state, shape (batch_size, 1).
        """
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.policy_fc(p), dim=1).exp()

        return p, v


class Connect4Net(nn.Module):
    """
    Connect4Net is a neural network architecture designed for the game Connect 4.
    Args:
        num_residual_layers (int): Number of residual layers to include in the network. Default is 19.
    Attributes:
        initial_layer (InitialConvLayer): The initial convolutional layer that processes the input.
        residual_layers (nn.ModuleList): A list of residual layers for deep feature extraction.
        output_layer (OutputLayer): The final output layer producing the network's predictions.
    Methods:
        forward(x):
            Defines the forward pass of the network. Processes input `x` through the initial layer,
            followed by a sequence of residual layers, and finally through the output layer.
    Usage:
        model = Connect4Net(num_residual_layers=19)
        output = model(input_tensor)
    """
    def __init__(self, num_residual_layers=19):
        """
        Initializes the neural network model for AlphaZero.

        Args:
            num_residual_layers (int, optional): Number of residual layers to include in the network. Defaults to 19.

        Attributes:
            initial_layer (InitialConvLayer): The initial convolutional layer of the network.
            residual_layers (nn.ModuleList): A list of residual layers for deep feature extraction.
            output_layer (OutputLayer): The output layer producing policy and value predictions.
        """
        super().__init__()
        self.initial_layer = InitialConvLayer()
        self.residual_layers = nn.ModuleList([ResidualLayer() for _ in range(num_residual_layers)])
        self.output_layer = OutputLayer()

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the initial layer, residual layers, and output layer.
        """
        x = self.initial_layer(x)
        for layer in self.residual_layers:
            x = layer(x)
        return self.output_layer(x)


class CustomLoss(nn.Module):
    """
    CustomLoss is a PyTorch module that implements a combined loss function for training neural networks in reinforcement learning or game-playing scenarios, such as AlphaZero.
    This loss function consists of two components:
    1. Value Loss: Measures the mean squared error (MSE) between the predicted value and the target value.
    2. Policy Loss: Measures the cross-entropy loss between the predicted policy distribution and the target policy distribution.
    The forward method computes both losses and returns their mean sum as the final loss value.
        target_value (torch.Tensor): Ground truth value tensor.
        predicted_value (torch.Tensor): Predicted value tensor from the network.
        target_policy (torch.Tensor): Ground truth policy tensor (probability distribution).
        predicted_policy (torch.Tensor): Predicted policy tensor (probability distribution) from the network.
        torch.Tensor: The mean combined loss consisting of value loss (MSE) and policy loss (cross-entropy).
    """
    def __init__(self):
        """
        Initializes the instance of the class and calls the initializer of the parent class.
        """
        super().__init__()

    def forward(self, target_value, predicted_value, target_policy, predicted_policy):
        """
        Computes the combined loss for value and policy predictions.

        Args:
            target_value (torch.Tensor): The ground truth value tensor.
            predicted_value (torch.Tensor): The predicted value tensor from the network.
            target_policy (torch.Tensor): The ground truth policy tensor (probability distribution).
            predicted_policy (torch.Tensor): The predicted policy tensor (probability distribution) from the network.

        Returns:
            torch.Tensor: The mean combined loss consisting of value loss (mean squared error) and policy loss (cross-entropy).
        """
        value_loss = (predicted_value - target_value) ** 2
        policy_loss = torch.sum(-predicted_policy * (1e-8 + target_policy).log(), dim=1)
        return (value_loss.view(-1) + policy_loss).mean()