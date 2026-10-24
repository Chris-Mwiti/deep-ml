import numpy as np 

def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """

    if len(latencies_ms) == 0:
        return dict([])

    arr = np.array(latencies_ms)
    mean = np.mean(arr)

    throughput = 1000/mean
    throughput = throughput.round(2)

    mean = mean.round(4)

    percentile_50 = np.percentile(arr, 50).round(4)
    percentile_95 = np.percentile(arr, 95).round(4)
    percentile_99 = np.percentile(arr, 99).round(4)

    return dict([("throughput_per_sec", throughput), ("avg_latency_ms", mean), ("p50_ms", percentile_50), ("p95_ms", percentile_95), ("p99_ms", percentile_99)])

    pass