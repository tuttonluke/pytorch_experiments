# %%
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
# %%
train_dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

batch_size = 4
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

example = next(iter(train_loader))
features, label = example
features = features.reshape(batch_size, -1)
# %%
class LinearRegression(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # randomly initialise required parameters
        self.linear_layer = torch.nn.Linear(784, 1)
        
    def forward(self, features):
        return self.linear_layer(features)
# %%
model = LinearRegression()
print(model(features))

