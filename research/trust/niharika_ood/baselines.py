def max_softmax_score(probs):
    return 1 - probs.max(axis=1)