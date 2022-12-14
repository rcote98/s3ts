from dataclasses import dataclass

@dataclass
class TaskParameters:

    """ Settings for the auxiliary tasks. """

    main: bool = True               # Main Label Classification
    main_weight: int = 3            # Main Label Classification Weight
    
    discrete_intervals: int = 5     # Discretization Intervals

    disc: bool = True               # Discretized Clasification
    disc_weight: int = 1            # Discretized Clasification Weight

    pred: bool = True               # Discretized Prediction
    pred_time: int = None           # Discretized Prediction Time (if None, then window_size is chosen)
    pred_weight: int = 1            # Discretized Prediction Weight

    areg_ts: bool = True            # Time-Series Autoregression
    areg_ts_weight: int = 1         # Time-Series Autoregression Weight

    areg_img: bool = False          # Similarity Frame Autoregression
    areg_img_weight: int = 1        # Similarity Frame Autoregression Weight

@dataclass
class AugProbabilities:
    """ Data augmentation probabilites. """
    jitter:      float = 0
    scaling:     float = 0
    time_warp:   float = 0
    window_warp: float = 0
