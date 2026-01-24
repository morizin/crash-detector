from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_metrics(metric_name: str):
    """Get the metric function based on the metric name.

    Args:
        metric_name (str): Name of the metric. Supported metrics are 'accuracy', 'precision', 'recall', 'f1'.

    Returns:
        function: Corresponding metric function.
    """
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }

    if metric_name not in metrics:
        raise ValueError(
            f"Unsupported metric: {metric_name}. Supported metrics are: {list(metrics.keys())}"
        )

    return metrics[metric_name]
