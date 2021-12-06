from abc import ABC, abstractmethod
from typing import List

from dicewars.ai.ladatron.map import Map


class Move:

    def do(self, map: Map):
        pass


class MoveGenerator(ABC):

    @abstractmethod
    def generate(self, player: int, map: Map) -> List[Move]:
        pass


class DumbMoveGenerator(MoveGenerator):

    def generate(self, player: int, map: Map) -> List[Move]:
        pass
