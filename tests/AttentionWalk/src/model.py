import torch

class AttentionWalkLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    For details see: https://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention
    """
    def __init__(self, args, shapes):
        """
        Setting up the layer.
        :param args: Arguments object.
        :param shapes: Shape of the target tensor.
        """
        super(AttentionWalkLayer, self).__init__()
        self.args = args
        self.shapes = shapes
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights.
        """
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], int(self.args.dimensions/2)))
        self.right_factors = torch.nn.Parameter(torch.Tensor(int(self.args.dimensions/2),self.shapes[1]))
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
        loss_on_matrices = torch.mean(torch.abs(self.args.num_of_walks*weighted_target_matrix.shape[0]*loss_on_target + loss_opposite))
        norms = torch.mean(torch.abs(self.left_factors))+torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.args.beta * (self.attention.norm(2)**2)
        loss = loss_on_matrices +  loss_on_regularization + self.args.gamma*norms
        return loss
