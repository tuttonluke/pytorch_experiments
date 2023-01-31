# %%
import numpy as np
import torch
import torch.nn.functional as F
# %%
X = torch.rand(300, 10)
print(type(X))

W = torch.randn(10)
b = torch.tensor([1])
# %% CASTING
array = np.random.randn(10, 5)
tensor = torch.from_numpy(array)
print(array.dtype, tensor.dtype)

# %% DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# %% AUTOMATIC DIFFERENTIATION
W = torch.randn(10, requires_grad=True) 
# only tensors of floating data can have a gradient
b = torch.tensor([1.], requires_grad=True)

y = X @ W + b
loss = y.sum()

print(W.grad, b.grad)

loss.backward()

print(W.grad, b.grad)

# %% CREATE MULTICLASS LOGISTIC REGRESSION FROM SCRATCH
class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.W = torch.nn.Parameter(torch.randn(self.n_features, self.n_classes))
        self.b = torch.nn.Parameter(torch.ones(self.n_classes))
    
    def forward(self, X):
        return X @ self.W + self.b
    
    def predict_proba(self, X):
        return torch.nn.functional.softmax(self(X), dim=1)
    
    def predict(self, X):
        return torch.argmax(self(X), dim=1)

# %% TENSORS
my_tensor = torch.randn(4, 3, 2)
print(my_tensor.shape)
