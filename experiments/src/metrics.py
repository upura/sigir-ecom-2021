import numpy as np
import torch
import torch.nn as nn


def f1_top_k(
    output: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 20,
) -> float:
    output_top_k = output.argsort(descending=True)[:, :top_k]
    all_f1 = []
    for p, l in zip(output_top_k, target):
        p = set(p.cpu().numpy())
        l = set(torch.where(l == 1)[0].cpu().numpy())   # onehot -> label
        nb_hits = len(p.intersection(l))
        precision = nb_hits / top_k
        recall = nb_hits / len(l) if len(l) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        all_f1.append(f1)
    f1_score = sum(all_f1) / len(target)
    return f1_score


def mrr_top_k(
    output: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 20,
) -> float:
    output_top_k = output.argsort(descending=True)[:, :top_k]
    rr = []
    for p, l in zip(output_top_k, target):
        p = list(p.cpu().numpy())
        l = list(torch.where(l == 1)[0].cpu().numpy())   # onehot -> label
        if len(l) == 0:
            rr.append(0.0)
        else:
            next_item = l[0]
            if next_item not in p:
                rr.append(0.0)
            else:
                rr.append(1.0 / (p.index(next_item) + 1))

    mrr = sum(rr) / len(target)
    return mrr


def evaluate_rec_task_metrics(
    output: torch.Tensor,
    target_next_item: torch.Tensor,
    target_subsequent_items: torch.Tensor,
    top_k: int = 20,
) -> dict:
    f1_score = f1_top_k(output, target_subsequent_items, top_k)
    mrr = mrr_top_k(output, target_next_item, top_k)
    metrics = {
        "f1_score": f1_score,
        "mrr": mrr,
    }
    return metrics
