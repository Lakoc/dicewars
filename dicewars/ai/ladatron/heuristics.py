from abc import ABC, abstractmethod

from dicewars.ai.ladatron.map import Map
import numpy as np


class Evaluation(ABC):

    @abstractmethod
    def evaluate(self, player: int, board_map: Map) -> float:
        pass


class HardcodedHeuristic(Evaluation):

    def evaluate(self, player: int, board_map: Map) -> float:
        player_areas = board_map.board_state[board_map.board_state[:, 0] == player]
        return 3 * player_areas.shape[0] + np.sum(player_areas[:, 1])
