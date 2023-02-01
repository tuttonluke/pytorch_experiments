# %%
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_diabetes
# %%
# train_dataset = MNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transforms.ToTensor()
# )

# batch_size = 4
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# example = next(iter(train_loader))
# features, label = example
# features = features.reshape(batch_size, -1)
class DiabetesDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X, self.y = load_diabetes(return_X_y=True)
    
    def __getitem__(self, index):
        return (torch.tensor(self.X[index]).float(), torch.tensor(self.y[index]).float())
    
    def __len__(self):
        return len(self.X)

class LinearRegression(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # randomly initialise required parameters
        self.linear_layer = torch.nn.Linear(10, 1)
        
    def forward(self, features):
        return self.linear_layer(features).reshape(-1)
# %%
dataset = DiabetesDataset()
train_loader = DataLoader(dataset, shuffle=True, batch_size=4)
model = LinearRegression()

def train(model, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    # initialise tensorboard visualiser
    writer = SummaryWriter()
    batch_index = 0

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward() # populates grad attributes
            print(loss.item())
            optimiser.step() # optimisation step
            optimiser.zero_grad() # reset gradients
            writer.add_scalar(tag="Loss", scalar_value=loss.item(), global_step=batch_index)
            batch_index += 1

train(model)
