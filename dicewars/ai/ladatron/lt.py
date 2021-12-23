import logging
import time

from dicewars.ai.ladatron.brs import BestReplySearch
from dicewars.ai.ladatron.heuristics import HardcodedHeuristic
from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.move_generators import FilteringMoveGenerator, LessDumbMoveGenerator, Move
from dicewars.ai.ladatron.moves import BattleMove, EndMove, MoveSequence, TransferMove
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board


class AI:

    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.max_transfers = max_transfers
        self.max_battles = 15

        self.depth = 3
        self.opponents = list(filter(lambda x: x != self.player_name, players_order))

        heuristic = HardcodedHeuristic()
        self.search = BestReplySearch(heuristic, FilteringMoveGenerator(heuristic), self.max_transfers, self.max_battles)
        self.moves: MoveSequence = MoveSequence()

        # TODO: Precompute as many moves as possible. We got 10 seconds in the constructor.

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        if len(self.moves) == 0:
            start_time = time.time()
            remaining_transfers = self.max_transfers - nb_transfers_this_turn
            remaining_attacks = self.max_battles - nb_moves_this_turn + nb_transfers_this_turn

            board_map: Map = Map.from_board(board)
            self.moves = self.search.search(self.player_name, self.opponents, board_map, depth=self.depth,
                                            remaining_transfers=remaining_transfers,
                                            remaining_attacks=remaining_attacks)
            self.logger.info(F"Number of moves found: {len(self.moves)}")
            self.logger.info(F"Time it took to compute the moves: {time.time() - start_time}")

        move: Move = self.moves.pop(0)
        while not self._is_valid_move(move, board):
            move: Move = self.moves.pop(0)
        return self._apply_move(move)

    def _is_valid_move(self, move: Move, board: Board) -> bool:
        if isinstance(move, BattleMove):
            source_area: Area = board.areas[str(move.source + 1)]
            target_area: Area = board.areas[str(move.target + 1)]

            return source_area.can_attack() and \
                   source_area.owner_name == self.player_name and \
                   target_area.owner_name != self.player_name
        elif isinstance(move, TransferMove):
            source_area: Area = board.areas[str(move.source + 1)]
            target_area: Area = board.areas[str(move.target + 1)]

            return source_area.owner_name == self.player_name and \
                   target_area.owner_name == self.player_name
        else:
            return True

    def _apply_move(self, move: Move) -> object:
        if isinstance(move, TransferMove):
            return TransferCommand(int(move.source + 1), int(move.target + 1))
        elif isinstance(move, BattleMove):
            return BattleCommand(int(move.source + 1), int(move.target + 1))
        elif isinstance(move, EndMove):
            return EndTurnCommand()
