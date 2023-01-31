# %%
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.datasets import load_diabetes

# %%
transfrom = transforms.PILToTensor()

dataset = MNIST(root="./data", download=True, train=True, transform=transfrom)

# %%
# dataloader allows making batches and shuffling data really easy
train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

for batch in train_loader:
    print(batch)
# %% custom torch dataset from sklearn dataset
class DiabetesDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X, self.y = load_diabetes(return_X_y=True)
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)

dataset = DiabetesDataset()
print(dataset[10])
len(dataset)
# %%
class CustomDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        assert len(X) == len(y) # feature and label arrays must have same length
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

# %% DataLoaders
dataset = CustomDataset(torch.randn(300, 10), torch.randint(0, 5, size=(300, )))
dataloader = DataLoader(dataset, batch_size=64)

for (X, y) in dataloader:
    print(X.shape, y.shape)