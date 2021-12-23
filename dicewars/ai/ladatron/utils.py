import numpy as np


def border_distance(player, board_map, max_depth):
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
