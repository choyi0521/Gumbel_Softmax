import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import save_image

epochs = 10
batch_size = 64
tau = 1.0
hard = True
n_vars = 20
n_classes = 10
print_freq = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

def gumbel_softmax_sample(logits, tau, eps=1e-20):
    u = torch.rand(logits.shape, device=logits.get_device())
    g = -torch.log(-torch.log(u + eps) + eps)
    x = logits + g
    return F.softmax(x / tau, dim=-1)

def gumbel_softmax(logits, tau, hard=False):
    y = gumbel_softmax_sample(logits, tau)
    if not hard:
        return y

    n_classes = y.shape[-1]
    z = torch.argmax(y, dim=-1)
    z = F.one_hot(z, n_classes)
    z = (z - y).detach() + y
    return z

class GumbelVAE(nn.Module):
    def __init__(self):
        super(GumbelVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, n_vars * n_classes)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_vars * n_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)
        logits = x.view(x.shape[0], n_vars, n_classes)
        q = F.softmax(logits, dim=-1).view(x.shape[0], -1)
        x = self.decode(gumbel_softmax(logits, tau, hard).view(x.shape[0], -1))
        return x, q

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

def compute_loss(pred, target, q):
    rc_loss = F.binary_cross_entropy(pred, target, reduction='sum') / target.shape[0]
    kl = (q * torch.log(q * n_classes)).sum(dim=-1).mean()
    loss = rc_loss + kl
    return loss



model = GumbelVAE().to(device)
optimizer = Adam(model.parameters())

for epoch in range(epochs):
    # training
    for iteration, (data, _) in enumerate(dataloader):
        data = data.to(device).view(-1, 784)

        optimizer.zero_grad()
        pred, q = model(data)
        loss = compute_loss(pred, data, q)
        loss.backward()
        optimizer.step()

        if iteration % print_freq == 0 or iteration == len(dataloader) - 1:
            print('Epoch[{0}]({1} / {2}) - Loss: {3}'.format(
                epoch + 1,
                iteration + 1,
                len(dataloader),
                loss.item()
            ))

    # sampling
    z = torch.randint(0, n_classes, (batch_size, n_vars), device=device)
    z = F.one_hot(z, n_classes).view(batch_size, -1)
    x = model.decode(z.to(torch.float))
    save_image(x.view(batch_size, 1, 28, 28), './data/sample_{0}.png'.format(epoch + 1))