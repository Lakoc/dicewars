from dicewars.server.board import Board
from typing import Type, TypeVar
import numpy as np

T = TypeVar('T', bound='Map')


class Map:
    def __init__(self, board_state, neighborhood_m):
        self.board_state = board_state
        self.neighborhood_m = neighborhood_m

    def copy(self):
        return Map(np.copy(self.board_state), self.neighborhood_m)

    @classmethod
    def from_board(cls: Type[T], board: Board) -> T:
        areas_count = len(board.areas)
        neighborhood_m = np.zeros((areas_count, areas_count), dtype=bool)
        for area in board.areas:
            np.put(neighborhood_m[int(area) - 1], np.array(board.areas[area].get_adjacent_areas_names()) - 1, 1)
        dice_counts = np.array([[board.areas[area].owner_name, board.areas[area].dice] for area in board.areas])
        return Map(dice_counts, neighborhood_m)
