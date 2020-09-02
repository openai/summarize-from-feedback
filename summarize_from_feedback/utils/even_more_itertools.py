from summarize_from_feedback.model_layout import ModelLayout


def distribute(iterable, layout: ModelLayout):
    """
    Of each group of layout.n_replicas successive items from the iterable, pick the one with index
    `layout.replica_idx`.
    Makes sure that the underlying iterator is advanced at the same pace no matter what replica_idx is.
    """
    it = iter(iterable)
    try:
        while True:
            for i in range(layout.replica_idx):
                next(it)
            ret = next(it)
            for i in range(layout.n_replicas - layout.replica_idx - 1):
                next(it)
            yield ret
    except StopIteration:
        return
