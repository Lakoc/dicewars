import logging

from dicewars.ai.ladatron.heuristics import HardcodedHeuristic
from dicewars.ai.ladatron.move_generators import DumbMoveGenerator
from dicewars.client.ai_driver import EndTurnCommand
from dicewars.ai.ladatron.brs import BestReplySearch

class AI:

    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.max_transfers = max_transfers

        self.search = BestReplySearch(HardcodedHeuristic(), DumbMoveGenerator())

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        oponnents = []
        player = 0
        self.search.search()

        return EndTurnCommand()
