import torch
import numpy as np

class AttentionWalkLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    For details see: https://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention
    """
    def __init__(self, embedd_dim, num_of_walks, beta, gamma, shapes):
        """
        Setting up the layer.
        :param embedd_dim: Number of dimensions. Default is 128.
        :param num_of_walks: Number of random walks. Default is 80.
        :param beta: Regularization parameter. Default is 0.5.
        :param gamma: Regularization parameter. Default is 0.5.
        :param shapes: Shape of the target tensor.
        """
        super(AttentionWalkLayer, self).__init__()
        self.embedd_dim = embedd_dim
        self.num_of_walks = num_of_walks
        self.beta = beta
        self.gamma = gamma
        self.shapes = shapes
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights.
        """
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], int(self.embedd_dim/2)))
        self.right_factors = torch.nn.Parameter(torch.Tensor(int(self.embedd_dim/2),self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0],1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.uniform_(self.left_factors,-0.01,0.01)
        torch.nn.init.uniform_(self.right_factors,-0.01,0.01)
        torch.nn.init.uniform_(self.attention,-0.01,0.01)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        """
        Doing a forward propagation pass.
        :param weighted_target_tensor: Target tensor factorized.
        :param adjacency_opposite: No-edge indicator matrix.
        :return loss: Loss being minimized.
        """
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim = 0)
        weighted_target_tensor = weighted_target_tensor * self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_matrix = torch.sum(weighted_target_tensor, dim=0).view(self.shapes[1],self.shapes[2])
        loss_on_target = - weighted_target_matrix * torch.log(torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_opposite = - adjacency_opposite * torch.log(1-torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_on_matrices = torch.mean(torch.abs(self.num_of_walks*weighted_target_matrix.shape[0]*loss_on_target + loss_opposite))
        norms = torch.mean(torch.abs(self.left_factors))+torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.beta * (self.attention.norm(2)**2)
        loss = loss_on_matrices +  loss_on_regularization + self.gamma*norms
        return loss


    def predict(self):
        # need tests
        x = torch.nn.functional.softmax(self.attention, dim = 0)
        return x