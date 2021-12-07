from math import inf
from typing import List

from dicewars.ai.ladatron.heuristics import Evaluation
from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.move_generators import MoveGenerator
from dicewars.ai.ladatron.moves import MoveSequence
from dicewars.ai.ladatron.turn import Turn


class BestReplySearch:

    def __init__(self, heuristic_evaluation: Evaluation, move_generator: MoveGenerator):
        self.heuristic_evaluation = heuristic_evaluation
        self.move_generator = move_generator

    def search(self, player: int, opponents: List[int], map: Map, depth: int) -> MoveSequence:
        if depth <= 0:
            raise ValueError("Unexpected depth")

        sequences: List[MoveSequence] = self.move_generator.generate_sequences(player, map)
        max_value: float = -inf
        best_sequence: MoveSequence = sequences[0]

        for sequence in sequences:
            map_new = sequence.do(map)
            value = self._search(player, opponents, map_new, depth=3, turn=Turn.MIN, alpha=-inf, beta=inf)

            if value > max_value:
                max_value = value
                best_sequence = sequence

        return best_sequence

    def _search(self, player: int, opponents: List[int], map: Map, depth: int, turn: Turn, alpha: float,
                beta: float) -> float:
        if depth <= 0:
            return self.heuristic_evaluation.eval(player, map)

        move_sequences: List[MoveSequence] = []
        if turn == Turn.MAX:
            move_sequences += self.move_generator.generate_sequences(player, map)
            turn = Turn.MIN
        else:
            for opponent in opponents:
                move_sequences += self.move_generator.generate_sequences(opponent, map)
            turn = Turn.MAX

        for sequence in move_sequences:
            map_new = sequence.do(map)
            value = -self._search(player, opponents, map_new, depth - 1, turn, -beta, -alpha)

            if value >= beta:
                return value
            alpha = max(alpha, value)

        return alpha
