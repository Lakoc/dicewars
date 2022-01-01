from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.ml.data import extract_features
from dicewars.ai.ladatron.ml.model import Network
from dicewars.ai.ladatron.utils import border_distance


class Evaluation(ABC):

    @abstractmethod
    def evaluate(self, player: int, board_map: Map) -> float:
        pass


class HardcodedDumbHeuristic(Evaluation):

    def evaluate(self, player: int, board_map: Map) -> float:
        player_areas = board_map.board_state[board_map.board_state[:, 0] == player]
        return 10 * player_areas.shape[0] + np.sum(player_areas[:, 1])


class HardcodedHeuristic(Evaluation):

    def __init__(self):
        self.distance_from_border_limit = 5

    def evaluate(self, player: int, board_map: Map) -> float:
        player_areas = board_map.board_state[board_map.board_state[:, 0] == player]
        return 4096 * player_areas.shape[0] + self.evaluate_transfer(player, board_map)

    def evaluate_transfer(self, player: int, board_map: Map) -> float:
        return weight_dice(player, board_map, self.distance_from_border_limit)

    def _size_of_largest_area(self, player: int, board_map: Map):
        """Find continuous area controlled by"""
        player_areas = board_map.board_state[:, 0] == player
        player_areas_neighbourhood = board_map.neighborhood_m * player_areas[:, np.newaxis] * player_areas[np.newaxis,
                                                                                              :]
        n_components, labels = connected_components(csgraph=csr_matrix(player_areas_neighbourhood), directed=False)
        players_labels = labels[player_areas]
        counts = np.bincount(players_labels)
        max_count = np.max(counts)
        return max_count


def weight_dice(player: int, board_map: Map, distance_from_border_limit: int):
    player_areas_mask = board_map.board_state[:, 0] == player
    player_areas_dice = board_map.board_state[player_areas_mask, 1]
    distances = border_distance(player, board_map, max_depth=distance_from_border_limit)
    distances[distances == -1] = distance_from_border_limit
    weights = distance_from_border_limit - distances[player_areas_mask]
    weighted_dice = np.sum(player_areas_dice * weights)
    # Return maximum possible weights if there is no neighbouring opponent
    # TODO: It could be moved directly to heuristic function. This is a side-effect.
    if weighted_dice == 0:
        full_dice = np.full_like(player_areas_dice, fill_value=8)
        weighted_dice = np.sum(full_dice * distance_from_border_limit)
    return weighted_dice


class NeuralNeuristic(Evaluation):

    def __init__(self):
        self.model = Network(input_features=5, output_features=1)
        self.model.load_state_dict(torch.load('dicewars/ai/ladatron/ml/models/model_99_end.weights'))
        self.distance_from_border_limit = 5

    def evaluate(self, player: int, board_map: Map) -> float:
        weighted_dice = weight_dice(player, board_map, self.distance_from_border_limit)
        with torch.no_grad():
            features = extract_features(player, [board_map], distance_from_border_limit=5)
            heuristic = self.model(torch.from_numpy(features).float())
            return 32768 * heuristic[0].numpy() + weighted_dice


def get_border_dices_diff(player, board_map):
    player_areas_mask = board_map.board_state[:, 0] == player
    opponent_areas_mask = board_map.board_state[:, 0] != player
    neighbourhood_with_opponents = board_map.neighborhood_m * player_areas_mask[:, np.newaxis] * \
                                   opponent_areas_mask[np.newaxis, :]
    player_border_areas, opponent_border_areas = np.where(neighbourhood_with_opponents)
    player_border_dice = np.sum(board_map.board_state[player_border_areas, 1])
    opponent_border_dice = np.sum(board_map.board_state[opponent_border_areas, 1])
    return player_border_dice - opponent_border_dice
