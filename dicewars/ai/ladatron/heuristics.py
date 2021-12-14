from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from dicewars.ai.ladatron.map import Map


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

    def evaluate(self, player: int, board_map: Map) -> float:
        player_areas = board_map.board_state[board_map.board_state[:, 0] == player]
        return 2 * player_areas.shape[0] + np.sum(player_areas[:, 1])

    def evaluate_transfer(self, player: int, board_map: Map) -> float:
        player_areas_mask = board_map.board_state[:, 0] == player
        opponent_areas_mask = board_map.board_state[:, 0] != player
        neighbourhood_with_opponents = board_map.neighborhood_m * player_areas_mask[:, np.newaxis] * opponent_areas_mask[np.newaxis, :]
        border_areas = np.where(neighbourhood_with_opponents)[0]
        border_dice = board_map.board_state[border_areas, 1]
        return np.sum(border_dice)

    def _size_of_largest_area(self, player: int, board_map: Map):
        """Find continuous area controlled by"""
        player_areas = board_map.board_state[:, 0] == player
        player_areas_neighbourhood = board_map.neighborhood_m * player_areas[:, np.newaxis] * player_areas[np.newaxis, :]
        n_components, labels = connected_components(csgraph=csr_matrix(player_areas_neighbourhood), directed=False)
        players_labels = labels[player_areas]
        counts = np.bincount(players_labels)
        max_count = np.max(counts)
        return max_count


if __name__ == "__main__":
    # A = np.array([[0, 0, 0, 0, 0],
    #               [0, 0, -5, -5, 0],
    #               [0, 1, 0, 1, 1],
    #               [0, 1, 1, 0, 1],
    #               [0, 0, 1, 1, 0]])
    # Maybe works
    # A = np.array([[1, 0, 0, 0, 0],
    #               [0, 1, -1, -1, 0],
    #               [0, 1, 1./3, 1./3, 1./3],
    #               [0, 1, 1./3, 1./3, 1./3],
    #               [0, 0, 1./3, 1./3, 1./3]])
    # A = np.array([[1, 0, 0, 0, 0],
    #               [0, 1, -10, -10, 0],
    #               [0, 10, 1, 1, 1],
    #               [0, 10, 1, 1, 1],
    #               [0, 0, 1, 1, 1]])
    # A = np.array([[10, 0, -1, 0],
    #               [0, 10, -1, 0],
    #               [1, 1, 1, 1],
    #               [0, 0, 1, 1]])
    # A = A / np.sum(A)
    # for i in range(10):
    #     A = np.linalg.matrix_power(A, 1)
    #     A / np.sum(A)
    # print(A)
    from scipy.sparse import csr_matrix

    adj = np.array([[0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 1]])

    num, labels = connected_components(csr_matrix(adj))
    counts = np.bincount(labels)
    pass
