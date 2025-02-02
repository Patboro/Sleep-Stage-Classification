import numpy as np

def split_signal_by_sleep_stage(signal, hypno, sleep_stage):
    stage_signal = []
    for index, value in enumerate(hypno):
        if value == sleep_stage:
            stage_signal.append(signal[index])
    
    stage_signal = np.array(stage_signal, dtype=float)

    return stage_signal
