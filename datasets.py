from typing import List

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, coeff: List[int], size: int) -> None:
        self.x = np.random.uniform(low=-1, high=1, size=size)
        self.data = np.polyval(coeff, self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[[idx]], self.data[[idx]]


class PiDataset(BaseDataset):

    def __init__(self, coeff: List[int], size: int) -> None:
        super().__init__(coeff=coeff, size=size)


class SigmaDataset(BaseDataset):

    def __init__(self, coeff: List[int], size: int) -> None:
        super().__init__(coeff=coeff, size=size)
        self.data = np.log(np.abs(self.data))
