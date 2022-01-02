from math import inf
from typing import List

from dicewars.ai.xpolok03.search.heuristics import Evaluation
from dicewars.ai.xpolok03.search.map import Map
from dicewars.ai.xpolok03.search.move_generators import MoveGenerator
from dicewars.ai.xpolok03.search.moves import MoveSequence
from dicewars.ai.xpolok03.search.turn import Turn


class BestReplySearch:

    def __init__(self, heuristic_evaluation: Evaluation, move_generator: MoveGenerator,
                 max_transfers: int, max_battles: int):
        self.heuristic_evaluation = heuristic_evaluation
        self.move_generator = move_generator
        self.max_transfers_per_turn = max_transfers
        self.max_battles_per_turn = max_battles

    def search(self, player: int, opponents: List[int], board_map: Map, depth: int,
               remaining_transfers: int, remaining_attacks: int) -> MoveSequence:
        if depth <= 0:
            raise ValueError("Unexpected depth")

        sequences: List[MoveSequence] = self.move_generator.generate_sequences(player, board_map, remaining_transfers,
                                                                               remaining_attacks)
        best_sequence: MoveSequence = sequences[0]

        if len(sequences) == 1:
            return best_sequence

        max_value = -inf
        for sequence in sequences:
            map_new = sequence.do(board_map)
            value = self._search(player, opponents, map_new, depth=depth - 1, turn=Turn.MIN, alpha=-inf, beta=inf)

            # Select the sequence that produces better score or
            # the longer sequence if scores are equal.
            if value > max_value or \
                    (value == max_value and len(sequence) > len(best_sequence)):
                max_value = value
                best_sequence = sequence

        return best_sequence

    def _search(self, player: int, opponents: List[int], board_map: Map, depth: int, turn: Turn, alpha: float,
                beta: float) -> float:
        if depth <= 0:
            return self.heuristic_evaluation.evaluate(player, board_map)  # + \
            # self.heuristic_evaluation.evaluate_transfer(player, board_map)

        max_alpha = alpha
        if turn == Turn.MAX:
            move_sequences: List[MoveSequence] = self.move_generator.generate_sequences(player, board_map,
                                                                                        self.max_transfers_per_turn,
                                                                                        self.max_battles_per_turn)
            max_alpha = max(max_alpha, self._do_moves(move_sequences, player, opponents, board_map,
                                                      depth, turn, alpha, beta))
        else:
            for opponent in opponents:
                move_sequences: List[MoveSequence] = self.move_generator.generate_sequences(opponent, board_map,
                                                                                            self.max_transfers_per_turn,
                                                                                            self.max_battles_per_turn)
                max_alpha = max(max_alpha, self._do_moves(move_sequences, player, opponents, board_map,
                                                          depth, turn, alpha, beta))
            max_alpha = -max_alpha
        return max_alpha

    def _do_moves(self, move_sequences, player: int, opponents: List[int], board_map: Map, depth: int, turn: Turn,
                  alpha: float, beta: float) -> float:
        for sequence in move_sequences:
            map_new = sequence.do(board_map)
            next_turn = turn.MAX if turn == turn.MIN else turn.MIN
            value = self._search(player, opponents, map_new, depth - 1, next_turn, alpha=-beta, beta=-alpha)
            if turn == turn.MAX:
                # if value > beta:
                #    return value
                alpha = max(alpha, value)  # (20 > -inf) => alpha = 20
            else:
                alpha = max(alpha, -value)
                if -value > beta:
                    return value
        return alpha
