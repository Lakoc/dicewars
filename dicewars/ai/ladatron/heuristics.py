from abc import ABC, abstractmethod

from dicewars.ai.ladatron.map import Map
import numpy as np


class Evaluation(ABC):

    @abstractmethod
    def eval(self, player: int, map: Map) -> float:
        pass


class HardcodedHeuristic(Evaluation):

    def eval(self, player: int, map: Map) -> float:
        player_areas = map.board_state[map.board_state[:, 0] == player]
        return 2 * player_areas.shape[0] + np.sum(player_areas[:, 1])
