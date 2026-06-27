# Merge json

import json

edge_json = json.load(open("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P1/refusal_scores_edge.json"))
mid_json = json.load(open("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P1/refusal_scores_mid.json"))


merged_json = {**edge_json, **mid_json}

with open("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/refusal_scores.json", "w") as f:
    json.dump(merged_json, f)


# Merge directions

import torch


directions_edge = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P1/directions_edge.pt")
directions_mid = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P1/directions_mid.pt")

merged_directions = {**directions_edge, **directions_mid}
torch.save(merged_directions, "/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/directions.pt")

# Merge best directions (Choose the best one)

best_direction_edge = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P1/best_direction_edge.pt")
best_direction_mid = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P1/best_direction_mid.pt")

best_layer_edge = best_direction_edge["layer"]
best_pos_edge = best_direction_edge["position"]

best_layer_mid = best_direction_mid["layer"]
best_pos_mid = best_direction_mid["position"]


score_edge = merged_json[f"({best_layer_edge}, {best_pos_edge})"]
score_mid = merged_json[f"({best_layer_mid}, {best_pos_mid})"]
best_direction = None
if score_edge > score_mid:
    best_direction = best_direction_edge
    print(f"best direction: edge | score: {score_edge}")
else:
    best_direction = best_direction_mid
    print(f"best direction: mid | score: {score_mid}")
torch.save(best_direction, "/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/best_direction.pt")