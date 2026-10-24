import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Your code here

    if len(latencies) == 0 :
        return dict([("P50", 0.0), ("P95", 0.0), ("P99", 0.0)])

    arr = np.array(latencies)

    percentile_50 = np.percentile(arr, 50).round(4)
    percentile_95 = np.percentile(arr, 95).round(4)
    percentile_99 = np.percentile(arr, 99).round(4)

    return dict([("P50", percentile_50), ("P95", percentile_95), ("P99", percentile_99)])

    pass