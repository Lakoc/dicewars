import collections
import random
from abc import ABC, abstractmethod
from typing import List

from dicewars.ai.ladatron.map import Map


class Move(ABC):

    @abstractmethod
    def do(self, board_map: Map):
        pass


class BattleMove(Move):

    def __init__(self, source, target, dice_difference=None):
        super().__init__()
        self.source = source
        self.target = target
        self.minimum_wear = 4
        self.dice_difference = dice_difference

    def do(self, board_map: Map):
        source_dice_count = board_map.board_state[self.source][1]
        target_dice_count = board_map.board_state[self.target][1]
        attack_power = random.randint(source_dice_count, source_dice_count * 6)
        def_power = random.randint(target_dice_count, target_dice_count * 6)

        if attack_power > def_power:
            board_map.board_state[self.target][0] = board_map.board_state[self.source][0]
            board_map.board_state[self.target][1] = board_map.board_state[self.source][1] - 1
            board_map.board_state[self.source][1] = 1
        else:
            board_map.board_state[self.target][1] -= board_map.board_state[self.source][1] // self.minimum_wear
            board_map.board_state[self.source][1] = 1

    def __lt__(self, other):
        return self.dice_difference < other.dice_difference

class TransferMove(Move):

    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target

    def do(self, board_map: Map):
        source_area_count_over = board_map.board_state[self.source][1] - 1
        board_map.board_state[self.source][1] = 1
        board_map.board_state[self.target][1] += source_area_count_over


class EndMove(Move):

    def __init__(self):
        super().__init__()

    def do(self, board_map: Map):
        pass


class MoveSequence(collections.Sequence):

    def __init__(self):
        super().__init__()
        self.moves: List[Move] = []

    def __getitem__(self, i: int) -> Move:
        return self.moves[i]

    def __len__(self):
        return len(self.moves)

    def pop(self, index: int):
        return self.moves.pop(index)

    def do(self, map: Map) -> Map:
        # Each sequence will be simulated in its own map.
        # This is to avoid move undoing if there is only a single map instance.
        map_copy = map.copy()
        for move in self.moves:
            move.do(map_copy)
        return map_copy

    def append(self, move: Move) -> None:
        self.moves.append(move)
