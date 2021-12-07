from abc import ABC, abstractmethod

from dicewars.ai.ladatron.map import Map
import random


class Move(ABC):

    @abstractmethod
    def do(self, board_map: Map):
        pass


class BattleMove(Move):

    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target
        self.minimum_wear = 4

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
