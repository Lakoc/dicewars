import collections
from abc import ABC, abstractmethod
from typing import _T_co, List, overload, Sequence

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


class MoveSequence(collections.Sequence[Move]):

    def __init__(self):
        super().__init__()
        self.moves: List[Move] = []

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> _T_co: ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> Sequence[_T_co]: ...

    def __getitem__(self, i: int) -> Move:
        return self.moves[i]

    def do(self, map: Map) -> Map:
        # Each sequence will be simulated in its own map.
        # This is to avoid move undoing if there is only a single map instance.
        map_copy = map.copy()
        for move in self.moves:
            move.do(map_copy)
        return map_copy

    def append(self, move: Move) -> None:
        self.moves.append(move)
