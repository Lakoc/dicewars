import json
import logging

import numpy as np

from agent.features import FeatureExtractor
from agent.policies.epsilon_greedy import epsilon_greedy
from agent.trainers.q_actor_critic_trainer import build_model
from agent.utils import read_iteration_from_file
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand


class AI:
    """Naive player agent

    This agent performs all possible moves in random order
    """

    def __init__(self, player_name, board, players_order, max_transfers):
        """
        Parameters
        ----------
        game : Game
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        with open('./agent/configs/qactorcritic.json', 'r') as config_file:
            self.config = json.load(config_file)
        arch_config = self.config['architecture']
        width = arch_config['matrix_width']

        # Load a model
        self.model = build_model(arch_config)
        self.model.load_weights('./agent/saved_models/actor.weights')

        # Setup a feature extractor
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.initialize(player_name, board, target_shape=[width, width])

        self.iteration = read_iteration_from_file(self.config)

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Get a random area. If it has a possible move, the agent will do it.
        If there are no more moves, the agent ends its turn.
        """
        features = self.feature_extractor.extract_features(board)
        features_batch = features[np.newaxis, ...]
        q_values = self.model(features_batch)[0]
        action = epsilon_greedy(q_values, features, self.iteration,
                                eps_min=self.config['greedy_policy']['eps_min'],
                                eps_max=self.config['greedy_policy']['eps_max'],
                                eps_decay_steps=self.config['greedy_policy']['eps_decay_steps'])

        if action is None:
            # No selected action
            return EndTurnCommand()
        else:
            # Correct indices (areas' name starts at 1)
            action += [1, 1, 0]
            src_area = board.get_area(action[0])
            dst_area = board.get_area(action[1])

            if action[2] == 0:
                # Attack command
                return BattleCommand(src_area.get_name(), dst_area.get_name())
            elif action[2] == 1:
                # Transfer dice command
                return TransferCommand(src_area.get_name(), dst_area.get_name())
            else:
                raise ValueError(F"Invalid value stored in 'action': {action}")
