import os
from datetime import datetime

import numpy as np
import torch

from dicewars.ai.ladatron.map import Map


def border_distance(player: int, board_map: Map, max_depth: int) -> np.ndarray:
    player_areas_mask = board_map.board_state[:, 0] == player
    opponent_areas_mask = board_map.board_state[:, 0] != player
    neighbourhood_with_opponents = board_map.neighborhood_m * player_areas_mask[:, np.newaxis] * \
                                   opponent_areas_mask[np.newaxis, :]
    player_border_areas, _ = np.where(neighbourhood_with_opponents)
    player_border_areas_mask = np.zeros(board_map.board_state.shape[0], dtype=bool)
    player_border_areas_mask[player_border_areas] = True

    dist = np.zeros(board_map.board_state.shape[0], dtype=int) - 1
    not_visited_areas = np.zeros(board_map.board_state.shape[0], dtype=bool)
    nodes_that_might_be_visited = player_areas_mask

    dist[player_border_areas] = 0
    areas_to_visit = player_border_areas

    for i in range(1, max_depth + 1):
        nodes_that_might_be_visited[areas_to_visit] = False
        if ~np.any(nodes_that_might_be_visited):
            break
        neighbours = np.argwhere(board_map.neighborhood_m[areas_to_visit] * nodes_that_might_be_visited)[:, 1]
        dist[neighbours] = i
        not_visited_areas[neighbours] = True
        areas_to_visit = neighbours

    return dist


def normalize_to_range(tensor, range, min_value=None, max_value=None):
    """
    Normalizes the tensor to the given range.
    The lower bound of the range is mapped to the minimal value of the given tensor.
    The uppoer bound of the range is mapped to the maximal value of the given tensor.
    For example:
    range = [0, 1]
    tensor = [4, 6, 8]
    normalized_tensor = [0, 0.5, 1]
    4 -> 0
    8 -> 1
    Parameters
    ----------
    tensor      A tensor to normalize.
    range       Expected range of values.
    Returns
    -------
    normalized_tensor
    """

    if min_value is None:
        min_value = np.min(tensor)
    else:
        min_value = np.array(min_value, dtype=tensor.dtype)
    if max_value is None:
        max_value = torch.max(tensor)
    else:
        max_value = np.array(max_value, dtype=tensor.dtype)

    tensor_zero_one = (tensor - min_value) / (max_value - min_value)

    range_width = range[1] - range[0]
    tensor_normalized = tensor_zero_one * range_width + range[0]
    return tensor_normalized


def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def make_timestamped_dir(path: str) -> str:
    """
    Creates a new directory with a name of a current timestamp
    in a location defined by 'path'.
    Parameters
    ----------
    path    string : Location of the new directory.
    Returns
    -------
    Returns path to the new directory.
    """
    timestamp = get_current_timestamp()
    subdir = os.path.join(path, timestamp)
    os.makedirs(subdir, exist_ok=True)
    return subdir
