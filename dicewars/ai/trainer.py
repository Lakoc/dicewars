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
        self.feature_extractor = FeatureExtractor(player_name, board, target_shape=[width, width])

        self.iteration = read_iteration_from_file(self.config)

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Get a random area. If it has a possible move, the agent will do it.
        If there are no more moves, the agent ends its turn.
        """
        try:
            features = self.feature_extractor.extract_features(board)
            # q_values = self.model(features)
            greedy_policy = self.config['greedy_policy']
            action = epsilon_greedy(features[:,:,0], self.feature_extractor.neighborhood_m, self.feature_extractor.owned_areas,
                                    self.iteration, eps_min=greedy_policy['eps_min'], eps_max=greedy_policy['eps_max'],
                                    eps_decay_steps=greedy_policy['eps_decay_steps'])
            # Correct indices (areas' name starts at 1)
            action += [1, 1, 0]

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
                src_area = board.areas[action[0]]
                dst_area = board.areas[action[1]]
                return TransferCommand(src_area.get_name(), dst_area.get_name())
            else:
                raise ValueError(F"Invalid value stored in 'action': {action}")
        except Exception as e:
            print('a')

    def select_valid_action(self, q_values, features):
        """
        Makes sure that the action is our to be made
        and that the areas are adjacent.
        """
        # Filter our moves
        our_moves_mask = features[:, :, 0:1]
        # Filter possible attacks
        attack_probs = features[:, :, 1:2]
        non_zero_attack_probs_mask = np.where(attack_probs > 0, 1, 0)
        masked_q_values = q_values * non_zero_attack_probs_mask * our_moves_mask

        argmax_q_value = np.argmax(masked_q_values[:, :, 0:1])
        if argmax_q_value[2] == 0:  # Attack
            return argmax_q_value
        else:  # Dice move
            # TODO
            raise NotImplementedError()
