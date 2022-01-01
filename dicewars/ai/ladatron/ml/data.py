import os
import pickle
from os import listdir
from os.path import isfile, join
from typing import List, Tuple
from torch.utils.data.dataset import random_split

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.utils import normalize_to_range


class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, path: str, dataset_type: str):
        self.path = path
        self.dataset_type = dataset_type

        file_paths = self._get_file_paths()
        train_proportion = 0.8
        files_count = len(file_paths)
        train_count = int(files_count * train_proportion)
        train_files, test_files = random_split(file_paths, [train_count, files_count - train_count])
        if dataset_type == 'train':
            self.features, self.heuristics = self._load_data(train_files)
        elif dataset_type == 'valid':
            self.features, self.heuristics = self._load_data(test_files)
        else:
            raise ValueError(F"Invalid dataset type: {dataset_type}")


    def _load_data(self, file_paths):
        players, winners, all_maps = self._load_file_contents(file_paths)
        all_features = []
        all_heuristics = []
        for player, winner, maps in zip(players, winners, all_maps):
            features = self._extract_features(player, winner, maps)
            all_features.append(features)
            heuristics = self._create_heuristic(player, winner, len(maps))
            all_heuristics.append(heuristics)

        x = np.concatenate(all_features)
        y = np.concatenate(all_heuristics)
        return x, y

    def _get_file_paths(self) -> List[str]:
        files = [join(self.path, file) for file in listdir(self.path)]
        files = [file for file in files if isfile(file)]
        return files

    def _load_file_contents(self, file_paths: List[str]) -> Tuple[List[int], List[int], List[Map]]:
        players = []
        winners = []
        all_maps = []

        for file_path in file_paths:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                continue
            with open(file_path, 'rb') as file:
                player, winner, maps = pickle.load(file)
            # heuristics = [state[0] for state in content]
            # maps = [state[1] for state in content]
            players.append(player)
            winners.append(winner)
            all_maps.append(maps)
        return players, winners, all_maps

    def _extract_features(self, player, winner, maps: List[Map]) -> np.ndarray:
        board_states = np.array([board_map.board_state for board_map in maps])
        neighborhoods = np.array([board_map.neighborhood_m for board_map in maps])

        max_areas = board_states.shape[1]
        max_dice = max_areas * 8

        player_areas_mask = board_states[:, :, 0] == player
        opponent_areas_mask = board_states[:, :, 0] != player

        player_areas = np.sum(player_areas_mask, axis=1)
        opponent_areas = np.sum(opponent_areas_mask, axis=1)

        player_areas_norm = normalize_to_range(player_areas, [-1, 1], min_value=0, max_value=max_areas)
        opponent_areas_norm = normalize_to_range(opponent_areas, [-1, 1], min_value=0, max_value=max_areas)

        player_dice_mask = board_states[:, :, 1] * player_areas_mask
        player_dice = np.sum(player_dice_mask, axis=1)
        opponent_dice_mask = board_states[:, :, 1] * opponent_areas_mask
        opponent_dice = np.sum(opponent_dice_mask, axis=1)

        player_dice_norm = normalize_to_range(player_dice, [-1, 1], min_value=0, max_value=max_dice)
        opponent_dice_norm = normalize_to_range(opponent_dice, [-1, 1], min_value=0, max_value=max_dice)

        pass

        features = np.stack([player_areas_norm, opponent_areas_norm,
                             player_dice_norm, opponent_dice_norm], axis=-1)
        return features

    def _create_heuristic(self, player: int, winner: int, length: int):
        if player == winner:
            return  np.full(length, fill_value=1)
        else:
            return np.full(length, fill_value=-1)
        # winner_space = np.linspace(0, 1, length)
        # looser_space = np.linspace(0, -1, length)
        # if player == winner:
        #     return winner_space
        # else:
        #     return looser_space

    def __len__(self):
        return self.heuristics.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.heuristics[idx]


def get_train_data_loader(batch_size=64):
    generator = DataGenerator('dataset', 'train')
    return DataLoader(generator, batch_size=batch_size, shuffle=True)


def get_test_data_loader(batch_size=64):
    generator = DataGenerator('dataset', 'valid')
    return DataLoader(generator, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    loader = get_test_data_loader(batch_size=4)
    features, heuristics = next(iter(loader))
    print(features.shape, heuristics.shape)
