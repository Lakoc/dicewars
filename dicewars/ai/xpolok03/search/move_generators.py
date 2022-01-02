from abc import ABC, abstractmethod
from enum import Enum
from math import inf
from typing import List

import numpy as np

from dicewars.ai.xpolok03.search.heuristics import Evaluation
from dicewars.ai.xpolok03.search.map import Map
from dicewars.ai.xpolok03.search.moves import BattleMove, EndMove, Move, MoveSequence, TransferMove
from dicewars.ai.xpolok03.search.utils import border_distance


class MovesType(Enum):
    TransferMoves = 1
    BattleMoves = 2,
    EndMoveOnly = 3


class MoveGenerator(ABC):

    def __init__(self, heuristic: Evaluation):
        self.heuristic = heuristic

    @abstractmethod
    def generate_moves(self, player: int, board_map: Map, transfers_allowed: bool, battles_allowed: bool) -> (
            List[Move], MovesType):
        pass

    @staticmethod
    def can_transfer_dices(board_map: Map, source_area: int, target_area: int):
        return board_map.board_state[source_area][1] > 1 and board_map.board_state[target_area][1] < 8

    def generate_sequences(self, player: int, board_map: Map, max_transfers, max_battles) -> List[MoveSequence]:
        moves, moves_type = self.generate_moves(player, board_map,
                                                transfers_allowed=(max_transfers > 0),
                                                battles_allowed=(max_battles > 0), transfer_priority=True)
        # if moves_type != MovesType.EndMoveOnly:
        #    moves.pop(0)  # Remove EndMove.
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

                next_move, last_value = self._search_moves(player, map_copy, last_value,
                                                           battles_remaining, transfers_remaining)
            sequence.append(next_move)
        return sequences

    def _search_moves(self, player, board_map, last_move_value, battles_remaining, transfers_remaining):
        # Generate possible moves. Either transfers or battles are returned.
        next_moves, moves_type = self.generate_moves(
            player, board_map,
            transfers_allowed=(transfers_remaining > 0),
            battles_allowed=(battles_remaining > 0))

        transfers_failed = False
        battles_failed = False

        while not transfers_failed or not battles_failed:
            if moves_type == MovesType.TransferMoves:
                next_move, next_value = self._select_best_move(player, next_moves, board_map,
                                                               self.heuristic.evaluate)
                if not isinstance(next_move, EndMove) and next_value > last_move_value:
                    return next_move, next_value

                # No transfer leads to a better state.
                # So try battles instead.
                transfers_failed = True
                if not battles_failed:
                    next_moves, moves_type = self.generate_moves(
                        player, board_map,
                        transfers_allowed=False,
                        battles_allowed=(battles_remaining > 0))

            if moves_type == MovesType.BattleMoves:
                next_move, next_value = self._select_best_move(player, next_moves, board_map,
                                                               self.heuristic.evaluate)
                if not isinstance(next_move, EndMove) and next_value > last_move_value:
                    return next_move, next_value

                # No battles leads to a better state so try transfers instead.
                battles_failed = True
                if not transfers_failed:
                    next_moves, moves_type = self.generate_moves(
                        player, board_map,
                        transfers_allowed=(transfers_remaining > 0),
                        battles_allowed=False)

            if moves_type == MovesType.EndMoveOnly:
                break
        return EndMove(), last_move_value

    def _select_best_move(self, player: int, moves: List[Move], board_map: Map, heuristic_func) -> (Move, float):
        best_move: Move = moves[0]
        best_value: float = -inf

        for move in moves:
            map_copy = board_map.copy()
            move.do(map_copy)
            value = heuristic_func(player, map_copy)
            if value > best_value:
                best_move = move
                best_value = value
        return best_move, best_value


class DumbMoveGenerator(MoveGenerator):

    def generate_moves(self, player: int, board_map: Map, transfers_allowed: bool, battles_allowed: bool) -> List[Move]:
        current_player_areas = np.argwhere(board_map.board_state[:, 0] == player).squeeze(axis=1)
        moves = []
        for source_area in current_player_areas:
            for neighbour_area in np.argwhere(board_map.neighborhood_m[source_area]).squeeze(axis=1):
                if transfers_allowed and neighbour_area in current_player_areas:
                    moves.append(TransferMove(source_area, neighbour_area))
                elif battles_allowed:
                    moves.append(BattleMove(source_area, neighbour_area))
        moves.append(EndMove())
        return moves


class LessDumbMoveGenerator(MoveGenerator):

    def generate_moves(self, player: int, board_map: Map, transfers_allowed: bool, battles_allowed: bool) -> List[Move]:
        current_player_areas = np.argwhere(board_map.board_state[:, 0] == player).squeeze(axis=1)
        moves = [EndMove()]
        for source_area in current_player_areas:
            for neighbour_area in np.argwhere(board_map.neighborhood_m[source_area]).squeeze(axis=1):
                is_neighbour_the_same_player = neighbour_area in current_player_areas
                can_transfer_dices = self.can_transfer_dices(board_map, source_area, neighbour_area)

                if transfers_allowed and is_neighbour_the_same_player and can_transfer_dices:
                    pass
                    # moves.append(TransferMove(source_area, neighbour_area))
                elif battles_allowed and self._is_reasonable_to_attack(board_map, source_area, neighbour_area):
                    moves.append(BattleMove(source_area, neighbour_area))
        return moves

    def _is_reasonable_to_attack(self, board_map, source_area, target_area):
        source_dice = board_map.board_state[source_area][1]
        target_dice = board_map.board_state[target_area][1]
        return (source_dice - target_dice) > 2 or source_dice == 8


class FilteringMoveGenerator(MoveGenerator):

    def __init__(self, heuristic: Evaluation, max_attacks=4, max_transfers=4, max_border_distance=4):
        super().__init__(heuristic)
        self.max_attacks = max_attacks
        self.max_transfers = max_transfers
        self.max_border_distance = max_border_distance

        self.battle_moves: List[BattleMove] = []
        self.transfer_moves: List[TransferMove] = []

    def generate_moves(self, player: int, board_map: Map, transfers_allowed: bool, battles_allowed: bool,
                       transfer_priority=True) -> (List[Move], MovesType):
        # Clear moves from last time
        self.battle_moves.clear()
        self.transfer_moves.clear()

        # First, find if there are any sensible transfers that can be made.
        if transfers_allowed:
            self.transfer_moves = self._generate_sensible_transfers(player, board_map)
            moves_type = MovesType.TransferMoves

        if (not transfer_priority or len(self.transfer_moves) == 0) and battles_allowed:
            moves_type = MovesType.BattleMoves
            # If there are no transfers, then resort to battles.
            self.battle_moves = self._generate_sensible_battles(player, board_map)

        # Filter moves further.
        self._filter_moves()

        # Put all moves together
        moves = [EndMove()]
        moves.extend(self.battle_moves)
        moves.extend(self.transfer_moves)
        if len(moves) == 1:
            moves_type = MovesType.EndMoveOnly
        return moves, moves_type

    def _generate_sensible_transfers(self, player: int, board_map: Map) -> List[TransferMove]:
        distance = border_distance(player, board_map, self.max_border_distance)
        player_areas_mask = board_map.board_state[:, 0] == player
        opponent_areas_mask = board_map.board_state[:, 0] != player

        transferable_areas = (board_map.board_state[:, 1] > 1) * player_areas_mask
        distance_to_border_mask = distance[np.newaxis, :] <= distance[:, np.newaxis]
        distance_to_border_mask[:, opponent_areas_mask] = False
        np.fill_diagonal(distance_to_border_mask, False)
        possible_transfers = distance_to_border_mask * transferable_areas[:, np.newaxis] * board_map.neighborhood_m

        dist_strength = np.max(distance) - distance
        sources, targets = np.where(possible_transfers)

        sources_dice = board_map.board_state[sources, 1]
        targets_dice = board_map.board_state[targets, 1]
        can_transfer_dice = sources_dice - 1
        can_receive_dice = 8 - targets_dice
        transfer_dice = np.minimum(can_transfer_dice, can_receive_dice)
        moves = []
        for i in range(len(targets)):
            transfer_size = transfer_dice[i]
            if transfer_size == 0:
                continue
            moves.append(TransferMove(sources[i], targets[i], transfer_size,
                                      transfer_size * dist_strength[targets[i]] / self.max_border_distance))
        return moves

    def _generate_sensible_battles(self, player: int, board_map: Map) -> List[BattleMove]:
        moves = []
        current_player_areas = np.argwhere(board_map.board_state[:, 0] == player).squeeze(axis=1)
        for source_area in current_player_areas:
            for neighbour_area in np.argwhere(board_map.neighborhood_m[source_area]).squeeze(axis=1):
                is_neighbour_the_same_player = neighbour_area in current_player_areas
                if not is_neighbour_the_same_player:
                    source_dice = board_map.board_state[source_area][1]
                    target_dice = board_map.board_state[neighbour_area][1]
                    dice_diff = source_dice - target_dice

                    if dice_diff >= 1 or source_dice == 8:
                        moves.append(BattleMove(source_area, neighbour_area, dice_diff))
        return moves

    def _filter_moves(self):
        # Select the first N best attacks
        self.battle_moves.sort(reverse=True)
        self.battle_moves = self.battle_moves[:self.max_attacks]

        # Select the best N transfers
        self.transfer_moves.sort(reverse=True)
        self.transfer_moves = self.transfer_moves[:self.max_transfers]

