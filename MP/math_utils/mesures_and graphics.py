def accuracy(target, s):
    predictions_idx = np.argmax(s, axis=-1)
    targets_idx = np.argmax(target, axis=-1)
    return np.mean(predictions_idx == targets_idx)
    # return predictions_idx, targets_idx