from torch.nn.functional import cross_entropy as torch_cross_entropy


def softmax_xent_loss_fn(outputs_mb, global_inputs_mb, reduction="mean", logits_key="logits"):
    """ Take a batch of logits and loss inputs and compute a scalar loss.

    If reduction="mean", average all losses.
    If reduction="none", return loss per token, with same shape is targets
    """
    targets = global_inputs_mb["targets"]
    flat_targets = targets.contiguous().view([-1])

    logits_mb = outputs_mb[logits_key]
    n_vocab = logits_mb.shape[-1]
    flat_logits = logits_mb.contiguous().view([-1, n_vocab])

    loss = torch_cross_entropy(
        input=flat_logits.float(), target=flat_targets.long(), reduction=reduction
    )
    if reduction == "mean":
        return loss
    return loss.reshape(targets.shape)
