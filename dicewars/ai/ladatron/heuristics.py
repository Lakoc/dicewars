from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.utils import border_distance


class Evaluation(ABC):

    @abstractmethod
    def evaluate(self, player: int, board_map: Map) -> float:
        pass

    @abstractmethod
    def evaluate_transfer(self, player: int, board_map: Map) -> float:
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
        # border_dice_diff = self._get_border_dices_diff(player, board_map)
        return 4096 * player_areas.shape[0] + self.evaluate_transfer(player, board_map)

    def _get_border_dices_diff(self, player, board_map):
        player_areas_mask = board_map.board_state[:, 0] == player
        opponent_areas_mask = board_map.board_state[:, 0] != player
        neighbourhood_with_opponents = board_map.neighborhood_m * player_areas_mask[:, np.newaxis] * \
                                       opponent_areas_mask[np.newaxis, :]
        player_border_areas, opponent_border_areas = np.where(neighbourhood_with_opponents)
        player_border_dice = np.sum(board_map.board_state[player_border_areas, 1])
        opponent_border_dice = np.sum(board_map.board_state[opponent_border_areas, 1])
        return player_border_dice - opponent_border_dice

    def evaluate_transfer(self, player: int, board_map: Map) -> float:
        player_areas_mask = board_map.board_state[:, 0] == player
        player_areas_dice = board_map.board_state[player_areas_mask, 1]
        distances = border_distance(player, board_map, max_depth=self.distance_from_border_limit)
        distances[distances == -1] = self.distance_from_border_limit
        weighted_players_dice = player_areas_dice * (self.distance_from_border_limit - distances[player_areas_mask])
        weighted_dice_total = np.sum(weighted_players_dice)
        return weighted_dice_total

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
