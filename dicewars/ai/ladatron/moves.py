from abc import ABC, abstractmethod

from dicewars.ai.ladatron.map import Map


class Move(ABC):

    @abstractmethod
    def do(self, map: Map):
        pass


class BattleMove(Move):

    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target

    def do(self, map: Map):
        pass


class TransferMove(Move):

    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target

    def do(self, map: Map):
        pass


class EndMove(Move):

    def __init__(self):
        super().__init__()

    def do(self, map: Map):
        pass
