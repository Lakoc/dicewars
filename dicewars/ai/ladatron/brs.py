import copy
from math import inf
from typing import List

from dicewars.ai.ladatron.heuristics import Evaluation
from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.move_generators import Move, MoveGenerator
from dicewars.ai.ladatron.turn import Turn


class BestReplySearch:

    def __init__(self, heuristic_evaluation: Evaluation, move_generator: MoveGenerator):
        self.heuristic_evaluation = heuristic_evaluation
        self.move_generator = move_generator

    def search(self, player: int, opponents: List[int], board_map: Map, depth: int) -> Move:
        if depth <= 0:
            raise ValueError("Unexpected depth")

        moves: List[Move] = self.move_generator.generate(player, board_map)
        max_value: float = -inf
        best_move: Move = moves[0]

        for move in moves:
            map_copy = board_map.copy()
            move.do(map_copy)
            value = self._search(player, opponents, board_map, depth=3, turn=Turn.MIN, alpha=-inf, beta=inf)

            if value > max_value:
                max_value = value
                best_move = move

        return best_move

    def _search(self, player: int, opponents: List[int], board_map: Map, depth: int, turn: Turn, alpha: float,
                beta: float) -> float:
        if depth <= 0:
            return self.heuristic_evaluation.eval(player, board_map)

        moves: List[Move] = []
        if turn == Turn.MAX:
            moves += self.move_generator.generate(player, board_map)
            turn = Turn.MIN
        else:
            for opponent in opponents:
                moves += self.move_generator.generate(opponent, board_map)
            turn = Turn.MAX

        for move in moves:
            # Each move will be simulated in its own board_map.
            # This is to avoid move undoing if there is only a single map instance.
            map_copy = board_map.copy()
            move.do(map_copy)
            value = -self._search(player, opponents, map_copy, depth - 1, turn, -beta, -alpha)

            if value >= beta:
                return value
            alpha = max(alpha, value)

        return alpha
