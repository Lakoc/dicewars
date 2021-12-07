from abc import ABC, abstractmethod
from math import inf
from typing import List

import numpy as np

from dicewars.ai.ladatron.heuristics import Evaluation
from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.moves import BattleMove, EndMove, Move, MoveSequence, TransferMove


class MoveGenerator(ABC):

    def __init__(self, heuristic: Evaluation):
        self.heuristic = heuristic

    @abstractmethod
    def generate_moves(self, player: int, board_map: Map, allow_transfers: bool, allow_battles: bool) -> List[Move]:
        pass

    @staticmethod
    def get_dice_diff(board_map: Map, area1: int, area2: int):
        return board_map.board_state[area1][1] - board_map.board_state[area2][1]

    @staticmethod
    def can_transfer_dices(board_map: Map, area1: int, area2: int):
        return board_map.board_state[area1][1] > 1 and board_map.board_state[area2][1] < 8

    def generate_sequences(self, player: int, board_map: Map, max_transfers, max_battles) -> List[MoveSequence]:
        moves: List[Move] = self.generate_moves(player, board_map,
                                                allow_transfers=(max_transfers > 0),
                                                allow_battles=(max_battles > 0))
        sequences: List[MoveSequence] = [MoveSequence() for _ in range(len(moves))]
        for move, sequence in zip(moves, sequences):
            # Set max transfers and battles for each sequence
            transfers_remaining = max_transfers
            battles_remaining = max_battles

            map_copy: Map = board_map.copy()
            next_move: Move = move
            last_value = -inf

            while not isinstance(next_move, EndMove):
                if isinstance(next_move, TransferMove):
                    transfers_remaining -= 1
                elif isinstance(next_move, BattleMove):
                    battles_remaining -= 1
                sequence.append(next_move)
                next_move.do(map_copy)
                next_moves: List[Move] = self.generate_moves(player, board_map,
                                                             allow_transfers=(transfers_remaining > 0),
                                                             allow_battles=(battles_remaining > 0))
                next_move, next_value = self._select_best_move(player, next_moves, map_copy)
                # If the heuristic says the previous state is no better than the proposed one.
                if last_value >= next_value:
                    break
                last_value = next_value
            sequence.append(EndMove())
        return sequences

    def _select_best_move(self, player: int, moves: List[Move], board_map: Map) -> (Move, float):
        best_move: Move = moves[0]
        best_value: float = -inf

        for move in moves:
            map_copy = board_map.copy()
            move.do(map_copy)
            value = self.heuristic.evaluate(player, map_copy)
            if value > best_value:
                best_move = move
                best_value = value
        return best_move, best_value


class DumbMoveGenerator(MoveGenerator):

    def generate_moves(self, player: int, board_map: Map, allow_transfers: bool, allow_battles: bool) -> List[Move]:
        current_player_areas = np.argwhere(board_map.board_state[:, 0] == player).squeeze(axis=1)
        moves = []
        for source_area in current_player_areas:
            for neighbour_area in np.argwhere(board_map.neighborhood_m[source_area]).squeeze(axis=1):
                if allow_transfers and neighbour_area in current_player_areas:
                    moves.append(TransferMove(source_area, neighbour_area))
                elif allow_battles:
                    moves.append(BattleMove(source_area, neighbour_area))
        moves.append(EndMove())
        return moves


class LessDumbMoveGenerator(MoveGenerator):

    def generate_moves(self, player: int, board_map: Map, allow_transfers: bool, allow_battles: bool) -> List[Move]:
        current_player_areas = np.argwhere(board_map.board_state[:, 0] == player).squeeze(axis=1)
        moves = []
        for source_area in current_player_areas:
            for neighbour_area in np.argwhere(board_map.neighborhood_m[source_area]).squeeze(axis=1):
                is_area_of_current_player = neighbour_area in current_player_areas
                can_transfer_dices = self.can_transfer_dices(board_map, source_area, neighbour_area)

                if allow_transfers and is_area_of_current_player and can_transfer_dices:
                    pass
                    # moves.append(TransferMove(source_area, neighbour_area))
                elif allow_battles and self.get_dice_diff(board_map, source_area, neighbour_area) > 0:
                    moves.append(BattleMove(source_area, neighbour_area))
        moves.append(EndMove())
        return moves
