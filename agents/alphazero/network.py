import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class BoardDataset(Dataset):
    """
    Custom PyTorch Dataset for Connect 4 board states, associated policy distributions, and value labels.
    Expects input data as a NumPy array with shape (N, 3), where each row contains:
      - board state (e.g., a NumPy array representing the game state),
      - policy (e.g., action probabilities),
      - value (e.g., game outcome or predicted value).
    """
    def __init__(self, data):
        """
        Initializes the dataset by unpacking the data into separate lists for states, policies, and values.

        Args:
            data (np.ndarray): A NumPy array of shape (N, 3)
        """  
        self.states = data[:, 0]
        self.policies = data[:, 1]
        self.values = data[:, 2]

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.states)

    def __getitem__(self, index):
        """
        Retrieves a sample at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): Board state tensor, transposed to shape (C, H, W) and converted to int64.
                - policy (np.ndarray): Action probabilities or logits.
                - value (float/int): Game outcome or predicted value.
        """
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

class OutputLayer(nn.Module):
    """
    Output layer with separate heads for policy and value.
    """
    def __init__( self,
        board_dims=(6, 7),
        policy_channels: int = 32,
        value_channels: int = 3
    ):
        """
        Initializes the neural network heads for value and policy prediction in a Connect4 AlphaZero agent.

        Args:
            board_dims (tuple, optional): Dimensions of the Connect4 board as (rows, columns). Defaults to (6, 7).
            policy_channels (int, optional): Number of output channels for the policy head convolution. Defaults to 32.
            value_channels (int, optional): Number of output channels for the value head convolution. Defaults to 3.

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
        # value head
        self.value_conv = nn.Conv2d(128, value_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_dims[0] * board_dims[1], 32)
        self.value_fc2 = nn.Linear(32, 1)

        # policy head
        self.policy_conv = nn.Conv2d(128, policy_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels, board_dims[1])

    def forward(self, x):
        """
        Performs a forward pass through the neural network, producing both policy and value outputs.

        Args:
            x (torch.Tensor): Input tensor representing the game state, typically of shape (batch_size, channels, height, width).

        Returns:
            tuple:
                - policy (torch.Tensor): Probability distribution over possible actions, obtained via softmax.
                - v (torch.Tensor): Scalar value prediction for each input in the batch, representing the expected outcome of the game.
        """
        # ---- Value head ----
        v = F.relu(self.value_bn(self.value_conv(x)))     
        v = v.view(v.size(0), -1)                          
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))                  

        # ---- Policy head ----
        p = F.relu(self.policy_bn(self.policy_conv(x)))     
        p = p.mean(dim=(2, 3))                             
        logits = self.policy_fc(p)                         
        policy = F.softmax(logits, dim=1)                  

        return policy, v
       
class SimpleConvBody(nn.Module):
    """
    Simpler convolutional stack.
    """
    def __init__(self, num_layers=5, channels=128):
        """
        Initializes a convolutional neural network stack with the specified number of layers and channels.

        Args:
            num_layers (int, optional): Number of convolutional layers in the stack. Defaults to 5.
            channels (int, optional): Number of input and output channels for each convolutional layer. Defaults to 128.

        Attributes:
            conv_stack (nn.Sequential): Sequential container of convolutional, batch normalization, and ReLU layers.
        """
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs a forward pass through the convolutional stack.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional stack.
        """
        return self.conv_stack(x)

class Connect4Net(nn.Module):
    """
    Connect4Net is a neural network architecture designed for the game Connect 4.
    
        num_conv_layers (int, optional): Number of convolutional layers in the convolutional body. Defaults to 5.

        conv_body (SimpleConvBody): The main convolutional body consisting of multiple convolutional layers.
        output_layer (OutputLayer): The final output layer producing the network's predictions (policy and value).

            Performs a forward pass through the network, processing input `x` through the initial layer,
            the convolutional body, and the output layer to produce policy and value predictions.
    """
    def __init__(self, num_conv_layers=5):
        """
        Initializes the neural network model for AlphaZero.
            num_conv_layers (int, optional): Number of convolutional layers to include in the network body. Defaults to 5.

            conv_body (SimpleConvBody): The main body of the network consisting of convolutional layers.
        """
        
        super().__init__()
        self.initial_layer = InitialConvLayer()
        self.conv_body = SimpleConvBody(num_layers=num_conv_layers)
        self.output_layer = OutputLayer()

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through.
        """
        x = self.initial_layer(x)
        x = self.conv_body(x)
        return self.output_layer(x)

class CustomLoss(nn.Module):
    """
    Custom loss combining MSE value loss and KL‐divergence policy loss
    in the AlphaZero style.
    """
    def __init__(self, value_coef: float = 0.4, policy_coef: float = 0.6):
        """
        Initializes the network with specified coefficients for value and policy losses.

        Args:
            value_coef (float, optional): Coefficient for the value loss component. Defaults to 0.4.
            policy_coef (float, optional): Coefficient for the policy loss component. Defaults to 0.6.
        """
        super().__init__()
        self.value_coef = value_coef
        self.policy_coef = policy_coef

    def forward(
        self,
        target_value: torch.Tensor,          
        predicted_value: torch.Tensor,      
        target_policy: torch.Tensor,         
        predicted_policy: torch.Tensor       
        ) -> torch.Tensor:
        """
        Computes the combined loss.

        Args:
            target_value (Tensor): True scalar values for each sample.
            predicted_value (Tensor): Predicted scalar values.
            target_policy (Tensor): True action probability distributions.
            predicted_policy (Tensor): Predicted action probability distributions.

        Returns:
            Tensor: Weighted sum of the value and policy loss.
        """
        # Value loss: MSE
        value_loss = F.mse_loss(predicted_value, target_value)

        # Policy loss: KL(target || pred)
        log_probs = torch.log(predicted_policy + 1e-8)
        policy_loss = F.kl_div(
            log_probs,           
            target_policy,       
            reduction="batchmean"
        )
        
        return self.value_coef * value_loss + self.policy_coef * policy_loss
