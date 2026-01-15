
import numpy as np
import pandas as pd

def calculate_cycles():
    # Synthetic parameters
    sampling_rate = 100 # Hz
    duration = 2 # seconds
    rpm = 300 # Revolutions per minute
    
    # Generate time
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate continuous angle (linear growth)
    freq = rpm / 60 # Hz (cycles per second)
    continuous_angle = 2 * np.pi * freq * t
    
    # Wrap to 0-2pi to simulate the sensor data
    crank_angle = continuous_angle % (2 * np.pi)
    
    df = pd.DataFrame({'Time': t, 'crank angle': crank_angle})
    
    # --- SOLUTION LOGIC ---
    # 1. Unwrap the phase to get cumulative angle
    # discon=2*np.pi ensures we catch the 2pi -> 0 wrap
    unwrapped_angle = np.unwrap(df['crank angle'].values, period=2*np.pi)
    
    # 2. Calculate cycle number (flooring the division by 2pi)
    # Adding a small epsilon to handle floating point issues at exactly 2pi multiples if needed, 
    # but generally floor is safe for "number of COMPLETE revolutions passed"
    df['Cycle number'] = np.floor(unwrapped_angle / (2 * np.pi)).astype(int)
    
    # Check results
    print(df.head(10))
    print("\n... wrapping point ...\n")
    
    # Find where cycle changes
    change_indices = np.where(np.diff(df['Cycle number']) > 0)[0]
    if len(change_indices) > 0:
        idx = change_indices[0]
        print(df.iloc[idx-2:idx+3])

if __name__ == "__main__":
    calculate_cycles()
