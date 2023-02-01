# %%
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_diabetes, make_classification, load_iris
# %%
class DiabetesDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X, self.y = load_diabetes(return_X_y=True)
    
    def __getitem__(self, index):
        return (torch.tensor(self.X[index]).float(), torch.tensor(self.y[index]).float())
    
    def __len__(self):
        return len(self.X)
    
class BinaryClassificationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X, self.y = make_classification(
            n_features=10, n_classes=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
        )
    
    def __getitem__(self, index):
        return (torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index].reshape(-1), dtype=torch.float32))
    
    def __len__(self):
        return len(self.X)

class IrisDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X, self.y = load_iris(return_X_y=True)
    
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

class LogisticRegression(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # randomly initialise required parameters
        self.linear_layer = torch.nn.Linear(10, 1)
        
    def forward(self, features):
        return torch.sigmoid(self.linear_layer(features))

class MulticlassClassification(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = torch.nn.Linear(4, 3)
    
    def forward(self, features):
        return F.softmax(self.linear_layer(features))
# %% 
def regression_train(model, epochs=10):
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

def binary_classification_train(model, epochs=10):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    # initialise tensorboard visualiser
    writer = SummaryWriter()
    batch_index = 0

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.binary_cross_entropy(prediction, labels)
            loss.backward() # populates grad attributes
            print(loss.item())
            optimiser.step() # optimisation step
            optimiser.zero_grad() # reset gradients
            writer.add_scalar(tag="Loss", scalar_value=loss.item(), global_step=batch_index)
            batch_index += 1

def multiclass_classification_train(model, epochs=10):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    # initialise tensorboard visualiser
    writer = SummaryWriter()
    batch_index = 0

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels.long())
            loss.backward() # populates grad attributes
            print(loss.item())
            optimiser.step() # optimisation step
            optimiser.zero_grad() # reset gradients
            writer.add_scalar(tag="Loss", scalar_value=loss.item(), global_step=batch_index)
            batch_index += 1
# %% REGRESSION EXAMPLE
dataset = DiabetesDataset()
train_loader = DataLoader(dataset, shuffle=True, batch_size=4)
model = LinearRegression()

regression_train(model)
# %% BINARY CLASSIFICATION EXAMPLE
dataset = BinaryClassificationDataset()
train_loader = DataLoader(dataset, shuffle=True, batch_size=4)
model = LogisticRegression()

binary_classification_train(model)
# %% MULTICLASS CLASSIFICATION EXAMPLE 
dataset = IrisDataset()
train_loader = DataLoader(dataset, shuffle=True, batch_size=4)
model = MulticlassClassification()

multiclass_classification_train(model)