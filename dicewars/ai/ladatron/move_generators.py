from abc import ABC, abstractmethod
from math import inf
from typing import List

import numpy as np

from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.moves import BattleMove, Move, MoveSequence, TransferMove


class MoveGenerator(ABC):

    @abstractmethod
    def generate_moves(self, player: int, board_map: Map) -> List[Move]:
        pass

    @staticmethod
    def get_dice_diff(board_map: Map, area1: int, area2: int):
        return board_map.board_state[area1][1] - board_map.board_state[area2][1]

    @staticmethod
    def can_transfer_dices(board_map: Map, area1: int, area2: int):
        return board_map.board_state[area1][1] > 1 and board_map.board_state[area2][1] < 8

    def generate_sequences(self, player: int, board_map: Map) -> List[MoveSequence]:
        moves: List[Move] = self.generate_moves(player, board_map)
        sequences: List[MoveSequence] = [MoveSequence() for _ in range(len(moves))]
        for move, sequence in zip(moves, sequences):
            map_copy: Map = board_map.copy()
            next_move: Move = move
            while next_move is not None:
                sequence.append(next_move)
                next_move.do(map_copy)
                next_moves: List[Move] = self.generate_moves(player, board_map)
                if len(next_moves) == 0:
                    next_move = None
                else:
                    next_move = self._select_best_move(next_moves, map_copy)
        return sequences

    def _select_best_move(self, moves: List[Move], board_map: Map) -> Move:
        best_move: Move = moves[0]
        best_value: float = -inf

        for move in moves:
            map_copy = board_map.copy()
            move.do(map_copy)
            value = self.heuristic.evaluate(map_copy)
            if value > best_value:
                best_move = move
                best_value = value
        return best_move


class DumbMoveGenerator(MoveGenerator):

    def generate_moves(self, player: int, board_map: Map) -> List[Move]:
        current_player_areas = np.argwhere(board_map.board_state[:, 0] == player).squeeze(axis=1)
        moves = [TransferMove(source_area, neighbour_area) if neighbour_area in current_player_areas else BattleMove(
            source_area, neighbour_area) for source_area in current_player_areas for neighbour_area in
                 np.argwhere(board_map.neighborhood_m[source_area]).squeeze(axis=1)]
        return moves


class LessDumbMoveGenerator(MoveGenerator):

    def generate_moves(self, player: int, board_map: Map) -> List[Move]:
        current_player_areas = np.argwhere(board_map.board_state[:, 0] == player).squeeze(axis=1)
        moves = []
        for source_area in current_player_areas:
            for neighbour_area in np.argwhere(board_map.neighborhood_m[source_area]).squeeze(axis=1):
                if neighbour_area in current_player_areas and self.can_transfer_dices(board_map, source_area,
                                                                                      neighbour_area):
                    moves.append(TransferMove(source_area, neighbour_area))
                elif self.get_dice_diff(board_map, source_area, neighbour_area) > 0:
                    moves.append(BattleMove(source_area, neighbour_area))
        return moves
