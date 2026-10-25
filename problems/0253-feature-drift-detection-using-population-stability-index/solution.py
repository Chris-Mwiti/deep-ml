import numpy as np

def detect_feature_drift(reference_data: list, production_data: list, num_bins: int = 10) -> dict:
    """
    Detect feature drift using Population Stability Index (PSI).
    
    Args:
        reference_data: List of feature values from reference distribution (e.g., training)
        production_data: List of feature values from production distribution
        num_bins: Number of bins for histogram comparison
    
    Returns:
        dict with 'psi', 'drift_detected', and 'drift_level'
    """

    #edge case 1
    if not reference_data or not production_data:
        return {}
    
    #edge case 2
    ref_arr = np.array(reference_data, dtype=float)
    prod_arr = np.array(production_data, dtype=float)

    #combine both the ref_arr & prod_arr for bininig

    combined = np.concatenate([ref_arr, prod_arr])

    min_val, max_val = np.min(combined), np.max(combined)

    #edge case 2
    if min_val == max_val:
        return {'psi': 0.0, 'drift_detected': False, 'drift_level': 'none'}

    #binning by histograms
    ref_count, _ = np.histogram(ref_arr, bins=num_bins, range=(min_val, max_val))

    prod_count, _ = np.histogram(prod_arr, bins=num_bins, range=(min_val, max_val))


    #convert counts into probabilites
    q = ref_count / len(ref_arr)
    p = prod_count / len(prod_arr)

    #handle zero's and re-normalize to maintain valid probability sums

    epsilon = 0.0001

    q = np.where(q == 0, epsilon, q)
    p = np.where(p == 0, epsilon, p)

    psi = np.sum((p - q) * np.log(p / q))
    psi_rounded = round(float(psi), 4)

    if psi_rounded < 0.1:
        drift_level = 'none'
        drift_detected = False
    elif psi_rounded < 0.25:
        drift_level = 'moderate'
        drift_detected = True
    else:
        drift_level = 'significant'
        drift_detected = True

    return {
        'psi': psi_rounded,
        'drift_detected': drift_detected,
        'drift_level': drift_level
    }

    pass