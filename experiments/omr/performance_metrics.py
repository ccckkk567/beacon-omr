"""Distinct timing scopes; merging validation cannot overwrite a train update."""


def round_performance(performance, split):
    prefix = 'perf/update/train/' if split == 'train' else f'perf/eval/round/{split}/'
    return {prefix + key: value for key, value in performance.items()}


def without_cached_round(metrics):
    # A cached final result is not another evaluation round.
    return {key: value for key, value in metrics.items() if not key.startswith('perf/eval/round/')}
