import torch.utils.data
from torch.utils.data import DataLoader


class DataGenerator(torch.utils.data.Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def get_train_data_loader(batch_size=64):
    generator = DataGenerator()
    return DataLoader(generator, batch_size=batch_size, shuffle=True)


def get_test_data_loader(batch_size=64):
    generator = DataGenerator()
    return DataLoader(generator, batch_size=batch_size, shuffle=False)
