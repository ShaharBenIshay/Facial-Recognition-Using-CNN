import random
import numpy as np
import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    # Define Model Architecture = init + forward methods
    def __init__(self, dropout_rate=0.0, use_gpu_flag=False):
        """
        Initialize the Siamese Network
        :param dropout_rate: the dropout rate for the dropout layer
        :param use_gpu_flag: boolean flag to indicate if the model should use GPU
        """
        super(SiameseNetwork, self).__init__()
        self.seed = 206201667  # Shahar's ID
        self.setup_seeds()
        self.model_use_gpu = use_gpu_flag
        self.dropout = dropout_rate
        self.device = torch.device("cuda") if (torch.cuda.is_available() and self.model_use_gpu) else torch.device(
            "cpu")
        self.conv_module = self._build_cnn_architecture().to(self.device)
        self.prediction_module = nn.Sequential(nn.Dropout(p=self.dropout),
                                               nn.Linear(4096, 1), nn.Sigmoid()).to(self.device)
        self.apply(self.setup_weights)  # Setup weights for each module in network

    def forward(self, input_1, input_2):
        """
        Forward pass of the Siamese Network
        :param input_1: 1st input to our CNN
        :param input_2: 2nd input to our CNN
        :return: prediction output of the Siamese Network between 0 and 1 (probability)
        """
        conv_1 = self.conv_module
        conv_2 = self.conv_module
        cnn_1_output = conv_1(input_1)
        cnn_2_output = conv_2(input_2)
        # The (last-1) layer - compute L1 distance with absolute distance, between siamese CNN twins
        L1_distance = torch.abs(cnn_1_output - cnn_2_output).float()  # size 4096 X 1
        prediction = self.prediction_module
        return prediction(L1_distance)

    def setup_seeds(self):
        """
        This function initializes the random number generators of PyTorch, Python's built-in random module, and NumPy
        with a specified seed value, in order to ensure that any results are reproducible
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def setup_weights(self, layer):
        """
        This function initializes the weights of the given layer
        (We tuned the weights initialization to ger better results, after some experiments were made)
        :param layer: torch.nn.Module - the layer to initialize its weights
        """
        if isinstance(layer, nn.Conv2d):
            w_mean, w_std = 0.0, 0.01
            nn.init.normal_(mean=w_std, std=w_mean, tensor=layer.weight)
            b_mean, b_std = 0.5, 0.01
            nn.init.normal_(mean=b_std, std=b_mean, tensor=layer.bias)

        if isinstance(layer, nn.Linear):
            w_mean, w_std = 0.0, 0.05
            nn.init.normal_(mean=w_std, std=w_mean, tensor=layer.weight)
            b_mean, b_std = 0.5, 0.05
            nn.init.normal_(mean=b_std, std=b_mean, tensor=layer.bias)

    def _build_cnn_architecture(self):
        """
        This function builds the architecture of the convolutional network in the Siamese Network,
        as described in the paper "Siamese Neural Networks for One-shot Image Recognition"
        :return: A PyTorch Sequential model, representing the convolutional Network
        """
        # From the paper, section 3.1. Model: "The model consists of a sequence of convolutional layers, each of which
        # uses a single channel with filters of varying size and a fixed stride of 1.
        # The number of convolutional filters is specified as a multiple of 16 to optimize performance.
        # The network applies a ReLU activation function to the output feature maps, optionally followed by maxpooling
        # with a filter size and stride of 2."

        conv_in_channels_layer_1, conv_out_channels_layer_1 = 1, 64  # ->
        conv_in_channels_layer_2, conv_out_channels_layer_2 = conv_out_channels_layer_1, 128  # ->
        conv_in_channels_layer_3, conv_out_channels_layer_3 = conv_out_channels_layer_2, 128  # ->
        conv_in_channels_layer_4, conv_out_channels_layer_4 = conv_out_channels_layer_3, 256  # ->
        # Number of neurons in dense = number of channels * filter_size = 256 * (4,4) = 256 * 4 * 4 = 4096
        dense_layer_size = conv_out_channels_layer_4 * 4 * 4
        flatten_layer_size = conv_out_channels_layer_4 * 6 * 6  # 256 * 6 * 6 = 9216
        # Activation function for first L-2 layers is ReLU
        # Default stride in nn.Conv2d is 1
        # input size to block #1:  105 * 105 * 1
        # input must be: [channels=1, height=105, width=105]
        # ------------------------------------ Block 1 ------------------------------------
        block1_conv = nn.Conv2d(conv_in_channels_layer_1, conv_out_channels_layer_1, kernel_size=(10, 10)) # 96 * 96 * 64
        block1_batch_norm = nn.BatchNorm2d(64) # 96 * 96 * 64
        block1_activation = nn.ReLU() # 96 * 96 * 64
        block1_max_pool = nn.MaxPool2d(2) # 48 * 48 * 64
        # ------------------------------------ Block 2 ------------------------------------
        block2_conv = nn.Conv2d(conv_in_channels_layer_2, conv_out_channels_layer_2, kernel_size=(7, 7)) # 42 * 42 * 128
        block2_batch_norm = nn.BatchNorm2d(128) # 42 * 42 * 128
        block2_activation = nn.ReLU() # 42 * 42 * 128
        block2_max_pool = nn.MaxPool2d(2) # 21 * 21 * 128
        # ------------------------------------ Block 3 ------------------------------------
        block3_conv = nn.Conv2d(conv_in_channels_layer_3, conv_out_channels_layer_3, kernel_size=(4, 4)) # 18 * 18 * 128
        block3_batch_norm = nn.BatchNorm2d(128) # 18 * 18 * 128
        block3_activation = nn.ReLU() # 18 * 18 * 128
        block3_max_pool = nn.MaxPool2d(2) # 9 * 9 * 128
        # ------------------------------------ Block 4 ------------------------------------
        block4_conv = nn.Conv2d(conv_in_channels_layer_4, conv_out_channels_layer_4, kernel_size=(4, 4)) # 6 * 6 * 256
        block4_batch_norm = nn.BatchNorm2d(256) # 6 * 6 * 256
        block4_activation = nn.ReLU() # 6 * 6 * 256
        # ------------------------------------ Flatten & Fully-Connected Block ------------------------------------
        flatten_layer = nn.Flatten() # 9216 * 1
        # flatten size = 9216, dense size = 4096
        dense_layer = nn.Linear(flatten_layer_size, dense_layer_size) # 4096 * 1
        sigmoid_layer = nn.Sigmoid() # 4096 * 1
        # ------------------------------------ Build Sequential ------------------------------------
        model = nn.Sequential(block1_conv, block1_batch_norm, block1_activation, block1_max_pool,
                              block2_conv, block2_batch_norm, block2_activation, block2_max_pool,
                              block3_conv, block3_batch_norm, block3_activation, block3_max_pool,
                              block4_conv, block4_batch_norm, block4_activation,
                              flatten_layer,
                              dense_layer,
                              sigmoid_layer)

        return model
