import numpy as np
import torch
from Levenshtein import distance
from nltk.translate.bleu_score import sentence_bleu

def eval_model(references, hypotheses):
    # Calculate scores
    bleu4 = 0.0
    for i, j in zip(references, hypotheses):
        bleu4 += max(sentence_bleu([i], j), 0.01)
    bleu4 = bleu4 / len(references)
    exact_match = exact_match_score(references, hypotheses)
    edit_distance = edit_distance_score(references, hypotheses)
    print((
        f"\n * BLEU-4:{bleu4:.4f}, Exact Match:{exact_match:.4f}, "
        f"Edit Distance:{edit_distance:.4f}"))
    return bleu4, exact_match, edit_distance


def exact_match_score(references, hypotheses):
    """Computes exact match scores.
    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)
    Returns:
        exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))

def edit_distance_score(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return d_leven / len_tot


def topk_acc(scores: torch.Tensor, targets: torch.Tensor, k: int):
    """
        from https://github.com/qs956/Latex_OCR_Pytorch
        LICENSE: AGPL-3.0 license 
        Computes top-k accuracy, from predicted and true labels.
        :param scores: scores from the model
        :param targets: true labels
        :param k: k in top-k accuracy
        :return: top-k accuracy
    """
    batch_size = targets.shape[0]
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


if __name__ == "__main__":
    # refs = [["a", "b", " ", "c", "d", ";", "e", "f", "g"], ["a", "b", " ", "c", "d", ";", "e", "f", "g"], ["a", "b", " ", "c", "d", ";", "e", "f", "g"]]
    hyps = [["a", "c", "b", "d", "e", "f", "g"],["2", "1", "3", "4", "5", "6", "7"]]
    refs = [["a", "b", "c", "d", "e", "f", "g"], ["1", '2', "3", "4" ,"5", "6", "7"]]
    # hyps = [["a c b d e f g"], ["2 1 3 4 5 6 7"]]
    eval_model(refs, hyps)
