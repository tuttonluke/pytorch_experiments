# %%
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch.nn as F
import torch
# %%
# convert raw data into a tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# download raw data and split into train and validation
train = datasets.MNIST("", train=True, 
                            transform=transform,
                            download=True)
train, val = random_split(train, [50000, 10000])

# initiate DataLoaders
train_loader = DataLoader(train, shuffle=True, batch_size=32)
val_loader = DataLoader(val, shuffle=True, batch_size=32)

# bulid the model
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        # self.fc1 = nn.Linear(28*28, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)
    
    def forward(self, X):
        X = X.view(X.shape[0], -1) # flatten the images
        return self.layers(X)

model = Network()

if torch.cuda.is_available():
    model = model.cuda()

# define criterion and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
writer = SummaryWriter()
# %% training loop
epochs = 5
batch_index = 0
min_val_loss = np.inf
min_val_epoch = 0

for epoch in range(epochs):
    train_loss = 0.0
    for features, labels in tqdm(train_loader):
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()
        # clear gradients
        optimiser.zero_grad()
        # forward pass
        prediction = model(features)
        # calculate loss
        loss = criterion(prediction, labels)
        # calculate gradients
        loss.backward()
        # update weights
        optimiser.step()
        # writer.add_scalar(tag="Training Loss", scalar_value=loss.item(), global_step=batch_index)
        batch_index += 1
        # calculate loss
        train_loss += loss.item()
    

    # calculate validation loss
    val_loss = 0.0
    batch_index = 0
    model.eval()
    for features, labels in val_loader:
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()
        target = model(features)
        loss = criterion(target, labels)
        val_loss += loss.item()
        # writer.add_scalar(tag="Validation Loss", scalar_value=loss.item(), global_step=batch_index)
        batch_index += 1
    
    if (epoch + 1) % 1 == 0:
        print(f"""Epoch: {epoch + 1} 
            Training Loss: {train_loss / len(train_loader):.4f}
            Validation Loss: {val_loss / len(val_loader):.4f}""")

    if min_val_loss > val_loss:
        min_val_loss = val_loss
        min_val_epoch = epoch + 1

print(f"""\nLowest validation loss is {min_val_loss / len(val_loader):.4f}
        at epoch {min_val_epoch}.""")






# %%
