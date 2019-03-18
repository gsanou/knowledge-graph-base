import time
import abc
import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from examples.pytorch.graphsage.graphsage import GraphSAGE
import dgl.function as fn


data = citegrh.load_cora()
features = torch.FloatTensor(data.features)
labels = torch.LongTensor(data.labels)
train_mask = torch.ByteTensor(data.train_mask)
val_mask = torch.ByteTensor(data.val_mask)
test_mask = torch.ByteTensor(data.test_mask)
in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()
print("""----Data statistics------'
  #Edges %d
  #Classes %d
  #Train samples %d
  #Val samples %d
  #Test samples %d""" %
      (n_edges, n_classes,
       train_mask.sum().item(),
       val_mask.sum().item(),
       test_mask.sum().item()))

# torch.set_device(torch.device('cuda'))
features = features.cuda()
labels = labels.cuda()
train_mask = train_mask.cuda()
val_mask = val_mask.cuda()
test_mask = test_mask.cuda()


# graph preprocess and calculate normalization factor
g = DGLGraph(data.graph)
n_edges = g.number_of_edges()

# create GraphSAGE model
model = GraphSAGE(g,
                  in_feats,
                  n_hidden=36,
                  n_classes = n_classes,
                  n_layers=2,
                  activation=F.relu,
                  dropout=0.5,
                  aggregator_type='pooling'
                  )

model.cuda()
loss_fcn = torch.nn.CrossEntropyLoss()

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# initialize graph
dur = []
for epoch in range(100):
    model.train()
    t0 = time.time()
    # forward
    logits = model(features)
    loss = loss_fcn(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(model, features, labels, val_mask)
    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
          "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                        acc, n_edges / np.mean(dur) / 1000))

print()
acc = evaluate(model, features, labels, test_mask)
print("Test Accuracy {:.4f}".format(acc))
