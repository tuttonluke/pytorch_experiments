# %%
from PIL import Image
import PIL
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pathlib
from torch.utils.data import Dataset
import typing

# %%
@dataclass
class DataGenerator:
    n_images: int
    n_classes: int

    def generate(self, path: pathlib.Path) -> pd.DataFrame:
        path.mkdir(parents=True, exist_ok=True)
        filenames = []
        for i in range(self.n_images):
            img_array = np.random.rand(100, 100, 3) * 255
            filename = (path / str(i)).with_suffix(".png")
            Image.fromarray(img_array.astype("uint8")).convert("RGBA").save(filename)

            filenames.append(filename)
        
        return pd.DataFrame(
            {
                "class" : np.random.randint(
                    0, high=self.n_classes, size=(self.n_images)
                ).tolist(),
            }
        )
# %%
class Files(Dataset):
    @classmethod
    def from_folder(
        cls, path: pathlib.Path, df: pd.DataFrame, transform=None, regex: str = "*"
    ) -> "Files":
        files = [file for file in path.glob(regex)]
        
        return cls(files, df, transform)
    
    def __init__(
        self, files: typing.List[pathlib.Path], df: pd.DataFrame, transform=None
    ):
        super().__init__()
        self.files = files
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        raise NotImplementedError


class Images(Files):
    def __getitem__(self, index) -> typing.Tuple[PIL.Image, int]:
        klass = self.df.iloc[index]
        sample = Image.open(self.files[index])
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, klass