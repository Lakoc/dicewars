from abc import ABC, abstractmethod
from typing import List

from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.moves import Move, TransferMove, BattleMove
import numpy as np


class MoveGenerator(ABC):

    @abstractmethod
    def generate(self, player: int, map: Map) -> List[Move]:
        pass

    @staticmethod
    def get_dice_diff(map: Map, area1: int, area2: int):
        return map.board_state[area1][1] - map.board_state[area2][1]

    @staticmethod
    def can_transfer_dices(map: Map, area1: int, area2: int):
        return map.board_state[area1][1] > 1 and map.board_state[area2][1] < 8


class DumbMoveGenerator(MoveGenerator):

    def generate(self, player: int, map: Map) -> List[Move]:
        current_player_areas = np.argwhere(map.board_state[:, 0] == player).squeeze(axis=1)
        moves = [TransferMove(source_area, neighbour_area) if neighbour_area in current_player_areas else BattleMove(
            source_area, neighbour_area) for source_area in current_player_areas for neighbour_area in
                 np.argwhere(map.neighborhood_m[source_area]).squeeze(axis=1)]
        return moves


class LessDumbMoveGenerator(MoveGenerator):

    def generate(self, player: int, map: Map) -> List[Move]:
        current_player_areas = np.argwhere(map.board_state[:, 0] == player).squeeze(axis=1)
        moves = []
        for source_area in current_player_areas:
            for neighbour_area in np.argwhere(map.neighborhood_m[source_area]).squeeze(axis=1):
                if neighbour_area in current_player_areas and self.can_transfer_dices(map, source_area, neighbour_area):
                    moves.append(TransferMove(source_area, neighbour_area))
                elif self.get_dice_diff(map, source_area, neighbour_area) > 0:
                    moves.append(BattleMove(source_area, neighbour_area))
        return moves
