import torch

change_scores = []
merged = {}

change_scores_0_6 = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P2/change_scores_0_6.pt")
change_scores_6_13 = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P2/change_scores_6_13.pt")
change_scores_13_20 = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P2/change_scores_13_20.pt")
change_scores_20_28 = torch.load("/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/P2/change_scores_20_28.pt")
change_scores.extend([change_scores_0_6, change_scores_6_13, change_scores_13_20, change_scores_20_28])

for change_score in change_scores:
    merged.update(change_score)
print(f"merged change_scores: {len(merged)}")

torch.save(merged, "/Users/erfan/Desktop/Thesis/from-neurons-to-directions/data/change_scores.pt")