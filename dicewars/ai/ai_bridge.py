from agent.agent import Agent


class AI:
    def __init__(self, player_name, board, players_order):
        self.agent = Agent(player_name, board, players_order)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        return self.agent.ai_turn(board, nb_moves_this_turn, nb_turns_this_game, time_left)
