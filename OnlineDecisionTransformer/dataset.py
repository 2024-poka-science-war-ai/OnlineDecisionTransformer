from torch.utils.data import Dataset, DataLoader

class trajDataset(Dataset):
    def __init__(self, K, t, R, s, a):
        super().__init__()
        self.K = K
        self.R = R
        self.s = s
        self.a = a
        #todo: implement custom dataset

    def __getitem__(self, index):

        return super().__getitem__(index)