from agent.src.ais import trainer


class AI:
    """
    A proxy to a trainer agent.
    """

    def __init__(self, player_name, board, players_order, max_transfers):
        self.agent = trainer.AI(player_name, board, players_order, max_transfers)

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        return self.agent.ai_turn(board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left)
