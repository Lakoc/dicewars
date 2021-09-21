import logging
from typing import Callable

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    """Naive player agent

    This agent performs all possible moves in random order
    """

    def __init__(self, player_name, board, players_order, turn_callback: Callable):
        """
        Parameters
        ----------
        game : Game
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        self.turn_callback = turn_callback

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Get a random area. If it has a possible move, the agent will do it.
        If there are no more moves, the agent ends its turn.
        """
        action = self.turn_callback(board)

        if action is None:
            # No selected action
            return EndTurnCommand()
        elif action[2] == 0:
            # Attack command
            from_area = board.areas[action[0]]
            at_area = board.areas[action[1]]
            return BattleCommand(from_area.get_name(), at_area.get_name())
        elif action[2] == 1:
            # Move dice command
            raise NotImplementedError()
            return EndTurnCommand()
        else:
            raise ValueError(F"Invalid value stored in 'action': {action}")
