import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from tqdm import tqdm

BATCH_SIZE=100
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        # z : 20d vector
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(model, optimizer):
    # set train mode
    model.train()
    epoch_loss = 0
    for i, (data, _) in enumerate(tqdm(train_loader)):
        data = data.cuda()
        # setup
        optimizer.zero_grad()
        # forward
        recon_image, mu, logsigma = model(data)
        # backward
        loss = loss_function(recon_image, data, mu, logsigma)
        epoch_loss += loss
        loss.backward()
        # optimize
        optimizer.step()

def evaluate(model):
    # set train mode
    model.eval()
    average_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.cuda()
        # forward
        recon_image, mu, logsigma = model(data)
        # backward
        loss = loss_function(recon_image, data, mu, logsigma)
        average_loss += loss

    print('Loss :', (average_loss/i).item())

def sample(model, tag=0):
    model.eval()
    # sample from normal
    #  20d vector
    z = torch.randn(BATCH_SIZE, 20).cuda()
    return model.decode(z).view(BATCH_SIZE, 1, 28, 28)
    #save_image(recon_image.view(BATCH_SIZE, 1, 28, 28), 'results/{}.png'.format(tag),
    #        nrow=BATCH_SIZE)


if __name__ == '__main__':
    model = VAE()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 30
    sampled_images = []
    for j in range(epochs):
        # save_image(torch.cat([data, recon_image.view(BATCH_SIZE, 1, 28, 28)]), 
        #        'image_{}.png'.format(j), nrow=BATCH_SIZE)
        print('EPOCH [{}]'.format(j))
        train(model, optimizer)
        evaluate(model)
        sampled_images.append(sample(model, tag=j))

    save_image(torch.cat(sampled_images), 'results/after_{}_epochs.png'.format(j+1), 
            nrow=BATCH_SIZE)
