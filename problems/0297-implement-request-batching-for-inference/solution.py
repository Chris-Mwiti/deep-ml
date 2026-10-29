import numpy as np

def batch_requests(requests: list, max_batch_size: int, max_wait_time: float) -> list:
    """
    Group inference requests into batches based on size and time constraints.
    
    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch
    
    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    # 1. Extract timestamps and sort indices using NumPy argsort
    timestamps = np.array([req['timestamp'] for req in requests])
    sorted_indices = np.argsort(timestamps)
    
    sorted_requests = [requests[i] for i in sorted_indices]
    
    batches = []
    current_ids = []
    current_features = []
    current_timestamps = []  # Keep track of timestamps in the current batch
    
    for req in sorted_requests:
        req_id = req['id']
        req_feat = req['features']
        req_time = req['timestamp']
        
        # Check constraints based on the *first* element's timestamp for wait time, 
        # or check against the current batch state. 
        # Here we check if adding it would exceed max_wait_time relative to the batch start.
        if current_timestamps:
            batch_start_time = current_timestamps[0]
            is_timed_out = (req_time - batch_start_time) >= max_wait_time
        else:
            is_timed_out = False
            
        is_full = len(current_ids) >= max_batch_size
        
        if is_full or is_timed_out:
            # Package the current batch using the MAX timestamp of the batch elements
            batches.append((
                current_ids,
                current_features,
                max(current_timestamps)  # <-- Getting the max timestamp here
            ))
            
            # Start a new batch
            current_ids = [req_id]
            current_features = [req_feat]
            current_timestamps = [req_time]
        else:
            current_ids.append(req_id)
            current_features.append(req_feat)
            current_timestamps.append(req_time)
            
    # Flush any remaining items at the end
    if current_ids:
        batches.append((
            current_ids,
            current_features,
            max(current_timestamps)  # <-- Getting the max timestamp here too
        ))
        
    return batches
    pass