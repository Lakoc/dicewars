import copy
from typing import List

from dicewars.ai.ladatron.heuristics import Evaluation
from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.move_generators import Move, MoveGenerator
from dicewars.ai.ladatron.turn import Turn


class BestReplySearch:

    def __init__(self, heuristic_evaluation: Evaluation, move_generator: MoveGenerator):
        self.heuristic_evaluation = heuristic_evaluation
        self.move_generator = move_generator

    def search(self, player: int, opponents: List[int], map: Map, depth: int, turn: Turn, alpha: float,
               beta: float) -> float:
        if depth <= 0:
            return self.heuristic_evaluation.eval(player, map)

        moves: List[Move] = []
        if turn == Turn.MAX:
            moves += self.move_generator.generate(player, map)
            turn = Turn.MIN
        else:
            for opponent in opponents:
                moves += self.move_generator.generate(opponent, map)
            turn = Turn.MAX

        for move in moves:
            # Each move will be simulated in its own map.
            # This is to avoid move undoing if there is only a single map instance.
            map_copy = copy.deepcopy(map)
            move.do(map_copy)
            value = -self.search(player, opponents, map_copy, depth - 1, turn, -beta, -alpha)

            if value >= beta:
                return value
            alpha = max(alpha, value)

        return alpha
