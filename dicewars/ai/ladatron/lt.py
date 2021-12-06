import logging

from dicewars.ai.ladatron.brs import BestReplySearch
from dicewars.ai.ladatron.heuristics import HardcodedHeuristic
from dicewars.ai.ladatron.map import Map
from dicewars.ai.ladatron.move_generators import DumbMoveGenerator, Move
from dicewars.ai.ladatron.moves import BattleMove, EndMove, TransferMove
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.board import Board


class AI:

    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.max_transfers = max_transfers

        self.search = BestReplySearch(HardcodedHeuristic(), DumbMoveGenerator())

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        # TODO: Initialize opponents and player's indices
        opponents = []
        player = 0

        map: Map = Map.from_board(board)
        move: Move = self.search.search(player, opponents, map, depth=3)

        return self._apply_move(move)

    def _apply_move(self, move: Move) -> object:
        if isinstance(move, TransferMove):
            return TransferCommand(move.source, move.target)
        elif isinstance(move, BattleMove):
            return BattleCommand(move.source, move.target)
        elif isinstance(move, EndMove):
            return EndTurnCommand()
