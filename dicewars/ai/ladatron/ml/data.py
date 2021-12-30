import pickle
from os import listdir
from os.path import isfile, join
from typing import List, Tuple

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from dicewars.ai.ladatron.map import Map


class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, path: str):
        self.path = path
        self.heuristics, self.features = self._load_data()

    def _load_data(self):
        file_paths = self._get_file_paths()
        heuristics, maps = self._load_file_contents(file_paths)
        features = self._extract_features(maps)
        return heuristics, features

    def _get_file_paths(self) -> List[str]:
        files = [join(self.path, file) for file in listdir(self.path)]
        files = [file for file in files if isfile(file)]
        return files

    def _load_file_contents(self, file_paths: List[str]) -> Tuple[List[int], List[Map]]:
        all_heuristics = []
        all_maps = []

        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                content = pickle.load(file)
            heuristics = [state[0] for state in content]
            maps = [state[1] for state in content]
            all_heuristics.extend(heuristics)
            all_maps.extend(maps)
        return all_heuristics, all_maps

    def _extract_features(self, map: List[Map]) -> np.ndarray:
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def get_train_data_loader(batch_size=64):
    generator = DataGenerator('dataset')
    return DataLoader(generator, batch_size=batch_size, shuffle=True)


def get_test_data_loader(batch_size=64):
    generator = DataGenerator('dataset')
    return DataLoader(generator, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    loader = get_test_data_loader(batch_size=4)
    features, heuristics = next(iter(loader))
    print(features.shape, heuristics.shape)
