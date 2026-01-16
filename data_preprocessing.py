"""
Seismic Data Preprocessing for Lithology Prediction

This script preprocesses post-stack seismic data for the GCDL (Geophysically 
Constrained Deep Learning) method. It generates training samples from well 
locations and prediction samples for the entire seismic volume.

Features:
- Multiple normalization modes (trace, RMS global, max global, none)
- Sliding window sample construction
- Support for multiple wells with different label formats

Author: GCDL Research Team
"""

import numpy as np
import pandas as pd
import os
import segyio
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional


# ======================= Configuration =======================
class Config:
    """Configuration parameters for data preprocessing."""
    
    # Input/Output paths (modify according to your data location)
    INPUT_PATH = "./data/raw"
    OUTPUT_PATH = "./data/processed"
    
    # Spatial parameters
    START_TIME = 3200  # Absolute start time (ms)
    FIRST_INLINE_ID = 2327
    FIRST_XLINE_ID = 3678
    INCREMENT = 2
    
    # Data dimensions
    N_INLINES = 151
    N_XLINES = 201
    EXPECTED_TRACES = N_INLINES * N_XLINES
    
    # Well coordinates (inline, xline)
    WELL_COORDS = {
        'well1': {'inline': 2467, 'xline': 3714},
        'well2': {'inline': 2589, 'xline': 3891},
        'well3': {'inline': 2391, 'xline': 4059}
    }
    
    # Seismic data file
    SEISMIC_FILE = 'seismic_poststack.sgy'
    
    # Sliding window parameters
    WINDOW_RADIUS = 14  # Window extends n samples up and down
    SAMPLING_INTERVAL = 4  # Sampling interval (ms)
    TARGET_TIME_RANGE = (3200, 3800)  # Target time range (ms)
    TIME_TOLERANCE = 2.5  # Time matching tolerance (ms)
    
    # Normalization mode: "none", "trace", "rms_global", "max_global"
    NORMALIZE_MODE = "trace"


def get_sample_length() -> int:
    """Calculate the sample length based on window radius."""
    return 2 * Config.WINDOW_RADIUS + 1


# ======================= Normalization Functions =======================
def normalize_trace(trace_data: np.ndarray) -> np.ndarray:
    """
    Trace-wise normalization: normalize each trace independently.
    
    Args:
        trace_data: 1D seismic trace data
    
    Returns:
        Normalized trace data
    """
    max_abs_value = np.abs(trace_data).max()
    return trace_data / (max_abs_value + 1e-10)


def normalize_rms_global(data_3d: np.ndarray) -> np.ndarray:
    """
    RMS global normalization: normalize entire 3D volume using global RMS.
    
    This is the recommended method as it preserves inter-trace amplitude 
    relationships and is robust to outliers.
    
    Args:
        data_3d: 3D seismic data (time, xline, inline)
    
    Returns:
        Normalized 3D data
    """
    print(f"\nApplying RMS global normalization...")
    print(f"Data shape: {data_3d.shape}")
    print(f"Original range: [{data_3d.min():.2e}, {data_3d.max():.2e}]")
    
    rms = np.sqrt(np.mean(data_3d ** 2))
    normalized = data_3d / (rms + 1e-10)
    
    print(f"Global RMS: {rms:.2e}")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    return normalized


def normalize_max_global(data_3d: np.ndarray) -> np.ndarray:
    """
    Global max normalization: normalize by dividing by maximum absolute value.
    
    Args:
        data_3d: 3D seismic data (time, xline, inline)
    
    Returns:
        Normalized 3D data with range [-1, 1]
    """
    print(f"\nApplying global max normalization...")
    print(f"Data shape: {data_3d.shape}")
    print(f"Original range: [{data_3d.min():.2e}, {data_3d.max():.2e}]")
    
    max_abs = np.max(np.abs(data_3d))
    normalized = data_3d / (max_abs + 1e-10)
    
    print(f"Global max absolute value: {max_abs:.2e}")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    return normalized


# ======================= Data I/O Functions =======================
def read_sgy_file(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read SEG-Y file and return seismic data.
    
    Args:
        filepath: Path to the SEG-Y file
    
    Returns:
        Tuple of (seismic_2d, time_samples) or (None, None) if failed
    """
    print(f"\nReading: {os.path.basename(filepath)}")
    
    try:
        with segyio.open(filepath, 'r', ignore_geometry=True, strict=False) as f:
            num_time = len(f.trace[0])
            total_traces = len(f.trace)
            
            print(f"File info: {num_time} time samples x {total_traces} traces")
            
            try:
                time_samples = f.samples
                print(f"Time range: {time_samples[0]:.1f} - {time_samples[-1]:.1f} ms")
            except:
                dt = 4.0
                time_samples = np.arange(num_time) * dt
                print(f"Using default time parameters: dt={dt}ms")
            
            seismic_2d = np.zeros((num_time, total_traces))
            for i in range(total_traces):
                seismic_2d[:, i] = f.trace[i]
            
            print(f"Data loaded: {seismic_2d.shape}")
            return seismic_2d, time_samples
            
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None, None


def reshape_to_3d(seismic_2d: np.ndarray, 
                  time_samples: np.ndarray) -> Tuple[Optional[np.ndarray], 
                                                      Optional[np.ndarray], 
                                                      Optional[np.ndarray], 
                                                      Optional[np.ndarray]]:
    """
    Reshape 2D seismic data to 3D (time x xline x inline).
    
    Args:
        seismic_2d: 2D seismic data (time x traces)
        time_samples: Time sample values
    
    Returns:
        Tuple of (data_3d, time_samples_abs, inlines, xlines)
    """
    num_time, total_traces = seismic_2d.shape
    print(f"\nReshaping to 3D: {seismic_2d.shape}")
    
    if total_traces != Config.EXPECTED_TRACES:
        print(f"Trace count mismatch: {total_traces} != {Config.EXPECTED_TRACES}")
        raise ValueError("Trace count mismatch")
    
    inlines = np.arange(Config.FIRST_INLINE_ID, 
                       Config.FIRST_INLINE_ID + Config.N_INLINES * Config.INCREMENT, 
                       Config.INCREMENT)
    xlines = np.arange(Config.FIRST_XLINE_ID, 
                      Config.FIRST_XLINE_ID + Config.N_XLINES * Config.INCREMENT, 
                      Config.INCREMENT)
    
    try:
        # Reshape: inline-major order then transpose
        temp = seismic_2d.reshape(num_time, Config.N_INLINES, Config.N_XLINES)
        data_3d = np.transpose(temp, (0, 2, 1))
        
        # Convert to absolute time if necessary
        if time_samples[0] < 1000:
            time_samples_abs = time_samples + Config.START_TIME
        else:
            time_samples_abs = time_samples
        
        print(f"3D reshape successful: {data_3d.shape}")
        print(f"Time range: {time_samples_abs[0]:.1f} - {time_samples_abs[-1]:.1f} ms")
        
        return data_3d, time_samples_abs, inlines, xlines
        
    except Exception as e:
        print(f"3D reshape failed: {e}")
        return None, None, None, None


# ======================= Label Processing =======================
def load_labels(well_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load label file for a specific well.
    
    Args:
        well_name: Name of the well (e.g., 'well1')
    
    Returns:
        Tuple of (target_times, target_labels)
    """
    print(f"\nLoading labels for {well_name}...")
    
    # Label file path (modify according to your data structure)
    label_file = os.path.join(Config.INPUT_PATH, f"{well_name}_labels.xlsx")
    
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        return None, None
    
    try:
        df = pd.read_excel(label_file)
        print(f"Label file shape: {df.shape}")
        
        times = df.iloc[:, 0].values
        labels = df.iloc[:, 1].values
        
        print(f"Time range: {times.min():.1f} - {times.max():.1f} ms")
        print(f"Label count: {len(labels)}")
        
        # Convert to absolute time if necessary
        if times.max() > 1000:
            input_times = times
        else:
            input_times = times + Config.START_TIME
        
        # Print label distribution
        unique_labels, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
        print(f"Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {int(label)}: {count}")
        
        return input_times, labels
        
    except Exception as e:
        print(f"Failed to load labels: {e}")
        return None, None


# ======================= Training Location Functions =======================
def get_well_training_locations(well_coords: Dict, 
                                inlines: np.ndarray, 
                                xlines: np.ndarray) -> List[Dict]:
    """
    Get training locations for a well (center trace and adjacent traces).
    
    Args:
        well_coords: Well coordinates dictionary
        inlines: Array of inline values
        xlines: Array of xline values
    
    Returns:
        List of location dictionaries
    """
    print(f"\nGetting training locations (Inline={well_coords['inline']}, Xline={well_coords['xline']})...")
    
    well_inline_idx = np.argmin(np.abs(inlines - well_coords['inline']))
    well_xline_idx = np.argmin(np.abs(xlines - well_coords['xline']))
    
    actual_inline = inlines[well_inline_idx]
    actual_xline = xlines[well_xline_idx]
    print(f"Actual well location: Inline={actual_inline:.0f}, Xline={actual_xline:.0f}")
    
    training_locations = []
    
    # Center trace and left/right adjacent traces
    offsets = [
        (0, 0, 'center'),
        (0, -1, 'left'),
        (0, 1, 'right')
    ]
    
    for inline_offset, xline_offset, position in offsets:
        new_inline_idx = well_inline_idx + inline_offset
        new_xline_idx = well_xline_idx + xline_offset
        
        if 0 <= new_inline_idx < len(inlines) and 0 <= new_xline_idx < len(xlines):
            training_locations.append({
                'inline_idx': new_inline_idx,
                'xline_idx': new_xline_idx,
                'inline': inlines[new_inline_idx],
                'xline': xlines[new_xline_idx],
                'position': position
            })
    
    print(f"Generated {len(training_locations)} training locations")
    
    return training_locations


# ======================= Sliding Window Sampling =======================
def sliding_window_sampling(data_trace: np.ndarray, 
                           time_samples: np.ndarray, 
                           center_time: float, 
                           center_label: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Generate a sliding window sample for a single trace.
    
    Args:
        data_trace: 1D time series data
        time_samples: Time sample values
        center_time: Center time point
        center_label: Label for center point
    
    Returns:
        Tuple of (sample, label) or (None, None) if invalid
    """
    n = Config.WINDOW_RADIUS
    interval = Config.SAMPLING_INTERVAL
    tolerance = Config.TIME_TOLERANCE
    
    # Find closest index to center time
    time_diff = np.abs(time_samples - center_time)
    center_idx = np.argmin(time_diff)
    
    if time_diff[center_idx] > tolerance:
        return None, None
    
    # Generate sample indices
    sample_indices = []
    valid_sample = True
    
    # Downward sampling
    for i in range(n, 0, -1):
        target_time = center_time - i * interval
        closest_idx = np.argmin(np.abs(time_samples - target_time))
        if abs(time_samples[closest_idx] - target_time) > interval * 2.0:
            valid_sample = False
            break
        sample_indices.append(closest_idx)
    
    # Center point
    if valid_sample:
        sample_indices.append(center_idx)
    
    # Upward sampling
    if valid_sample:
        for i in range(1, n + 1):
            target_time = center_time + i * interval
            closest_idx = np.argmin(np.abs(time_samples - target_time))
            if abs(time_samples[closest_idx] - target_time) > interval * 2.0:
                valid_sample = False
                break
            sample_indices.append(closest_idx)
    
    if not valid_sample or len(sample_indices) != 2 * n + 1:
        return None, None
    
    # Extract sample data
    sample_times = time_samples[sample_indices]
    sample_amplitudes = data_trace[sample_indices]
    
    sample = np.column_stack([sample_times, sample_amplitudes])
    
    return sample, center_label


# ======================= Sample Generation Functions =======================
def generate_training_samples(data_3d: np.ndarray, 
                             time_samples: np.ndarray, 
                             inlines: np.ndarray, 
                             xlines: np.ndarray, 
                             well_labels_dict: Dict) -> None:
    """
    Generate training samples for all wells.
    
    Args:
        data_3d: 3D seismic data
        time_samples: Time sample values
        inlines: Array of inline values
        xlines: Array of xline values
        well_labels_dict: Dictionary of well labels
    """
    print(f"\n{'='*60}")
    print(f"Generating training samples")
    print(f"Normalization mode: {Config.NORMALIZE_MODE}")
    print(f"{'='*60}")
    
    # Create output directory
    training_output_path = os.path.join(Config.OUTPUT_PATH, "training_samples")
    os.makedirs(training_output_path, exist_ok=True)
    
    all_samples = []
    total_samples = 0
    
    for well_name, well_coords in Config.WELL_COORDS.items():
        print(f"\nProcessing {well_name}...")
        
        if well_name not in well_labels_dict:
            print(f"Skipping {well_name} (no label data)")
            continue
        
        target_times, target_labels = well_labels_dict[well_name]
        training_locations = get_well_training_locations(well_coords, inlines, xlines)
        
        if not training_locations:
            print(f"Skipping {well_name} (no valid training locations)")
            continue
        
        well_samples = 0
        
        for loc in training_locations:
            inline_idx = loc['inline_idx']
            xline_idx = loc['xline_idx']
            
            trace_data = data_3d[:, xline_idx, inline_idx]
            
            # Apply trace normalization if selected
            if Config.NORMALIZE_MODE == "trace":
                trace_data = normalize_trace(trace_data)
            
            labeled_indices = np.where(~np.isnan(target_labels))[0]
            
            for label_idx in labeled_indices:
                center_time = target_times[label_idx]
                center_label = target_labels[label_idx]
                
                sample, label = sliding_window_sampling(
                    trace_data, time_samples, center_time, center_label
                )
                
                if sample is not None:
                    # Add location info: [time, amplitude, inline, xline, label]
                    sample_with_info = np.column_stack([
                        sample,
                        np.full(len(sample), loc['inline']),
                        np.full(len(sample), loc['xline']),
                        np.full(len(sample), np.nan)
                    ])
                    center_idx = len(sample) // 2
                    sample_with_info[center_idx, -1] = label
                    
                    all_samples.append(sample_with_info)
                    well_samples += 1
        
        print(f"{well_name} samples: {well_samples}")
        total_samples += well_samples
    
    print(f"\nTotal training samples: {total_samples}")
    
    # Save training samples
    if all_samples:
        suffix = get_normalization_suffix()
        output_file = os.path.join(training_output_path, f"training_samples{suffix}.txt")
        
        with open(output_file, 'w') as f:
            for sample in all_samples:
                np.savetxt(f, sample, fmt='%.6f')
                f.write('\n')
        
        print(f"Training samples saved: {output_file}")
        
        # Save sample info
        sample_info = []
        for i, sample in enumerate(all_samples):
            center_idx = len(sample) // 2
            center_row = sample[center_idx]
            sample_info.append({
                'sample_id': i,
                'time': center_row[0],
                'inline': center_row[2],
                'xline': center_row[3],
                'label': center_row[4]
            })
        
        info_df = pd.DataFrame(sample_info)
        info_file = os.path.join(training_output_path, f"training_info{suffix}.csv")
        info_df.to_csv(info_file, index=False)
        print(f"Sample info saved: {info_file}")


def generate_prediction_samples(data_3d: np.ndarray, 
                               time_samples: np.ndarray, 
                               inlines: np.ndarray, 
                               xlines: np.ndarray) -> None:
    """
    Generate prediction samples for the entire seismic volume.
    
    Args:
        data_3d: 3D seismic data
        time_samples: Time sample values
        inlines: Array of inline values
        xlines: Array of xline values
    """
    print(f"\n{'='*60}")
    print(f"Generating prediction samples")
    print(f"Normalization mode: {Config.NORMALIZE_MODE}")
    print(f"{'='*60}")
    
    # Create output directory
    prediction_output_path = os.path.join(Config.OUTPUT_PATH, "prediction_samples")
    os.makedirs(prediction_output_path, exist_ok=True)
    
    num_time, num_xlines, num_inlines = data_3d.shape
    print(f"Data dimensions: {num_time} time x {num_xlines} xline x {num_inlines} inline")
    
    # Filter target time range
    time_mask = ((time_samples >= Config.TARGET_TIME_RANGE[0]) & 
                 (time_samples <= Config.TARGET_TIME_RANGE[1]))
    valid_times = time_samples[time_mask]
    print(f"Valid time samples: {len(valid_times)}")
    
    all_samples = []
    total_samples = 0
    
    print(f"Processing {num_xlines} x {num_inlines} = {num_xlines * num_inlines} locations...")
    
    with tqdm(total=num_xlines * num_inlines, desc="Generating samples") as pbar:
        for xline_idx in range(num_xlines):
            for inline_idx in range(num_inlines):
                trace_data = data_3d[:, xline_idx, inline_idx]
                
                if Config.NORMALIZE_MODE == "trace":
                    trace_data = normalize_trace(trace_data)
                
                for center_time in valid_times:
                    sample, _ = sliding_window_sampling(
                        trace_data, time_samples, center_time, np.nan
                    )
                    
                    if sample is not None:
                        sample_with_info = np.column_stack([
                            sample,
                            np.full(len(sample), inlines[inline_idx]),
                            np.full(len(sample), xlines[xline_idx])
                        ])
                        
                        all_samples.append(sample_with_info)
                        total_samples += 1
                
                pbar.update(1)
    
    print(f"\nTotal prediction samples: {total_samples}")
    
    # Save prediction samples
    if all_samples:
        suffix = get_normalization_suffix()
        output_file = os.path.join(prediction_output_path, f"prediction_samples{suffix}.txt")
        
        print(f"Saving prediction samples...")
        with open(output_file, 'w') as f:
            for sample in tqdm(all_samples, desc="Saving"):
                np.savetxt(f, sample, fmt='%.6f')
                f.write('\n')
        
        print(f"Prediction samples saved: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        # Save sample info
        sample_info = []
        for i, sample in enumerate(all_samples):
            center_idx = len(sample) // 2
            center_row = sample[center_idx]
            sample_info.append({
                'sample_id': i,
                'time': center_row[0],
                'inline': center_row[2],
                'xline': center_row[3]
            })
        
        info_df = pd.DataFrame(sample_info)
        info_file = os.path.join(prediction_output_path, f"prediction_info{suffix}.csv")
        info_df.to_csv(info_file, index=False)
        print(f"Prediction info saved: {info_file}")


def get_normalization_suffix() -> str:
    """Get file suffix based on normalization mode."""
    suffix_map = {
        "none": "_raw",
        "trace": "_trace_norm",
        "rms_global": "_rms_global",
        "max_global": "_max_global"
    }
    return suffix_map.get(Config.NORMALIZE_MODE, "_raw")


# ======================= Main Function =======================
def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("Seismic Data Preprocessing for GCDL")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input path: {Config.INPUT_PATH}")
    print(f"  Output path: {Config.OUTPUT_PATH}")
    print(f"  Target time range: {Config.TARGET_TIME_RANGE[0]}-{Config.TARGET_TIME_RANGE[1]} ms")
    print(f"  Window radius: {Config.WINDOW_RADIUS}")
    print(f"  Sample length: {get_sample_length()}")
    print(f"  Normalization mode: {Config.NORMALIZE_MODE}")
    print("="*80)
    
    # Create output directories
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    # Step 1: Load well labels
    print("\n" + "="*80)
    print("Step 1: Loading well labels")
    print("="*80)
    
    well_labels_dict = {}
    for well_name in Config.WELL_COORDS.keys():
        target_times, target_labels = load_labels(well_name)
        if target_times is not None:
            well_labels_dict[well_name] = (target_times, target_labels)
        else:
            print(f"Warning: {well_name} labels not loaded")
    
    if not well_labels_dict:
        print("Error: No well labels loaded")
        return
    
    print(f"\nSuccessfully loaded labels for {len(well_labels_dict)} wells")
    
    # Step 2: Read seismic data
    print("\n" + "="*80)
    print("Step 2: Loading seismic data")
    print("="*80)
    
    filepath = os.path.join(Config.INPUT_PATH, Config.SEISMIC_FILE)
    seismic_2d, time_samples = read_sgy_file(filepath)
    
    if seismic_2d is None:
        print("Error: Failed to load seismic data")
        return
    
    # Step 3: Reshape to 3D
    data_3d, time_samples_abs, inlines, xlines = reshape_to_3d(seismic_2d, time_samples)
    
    if data_3d is None:
        print("Error: Failed to reshape data")
        return
    
    # Step 4: Apply global normalization if selected
    if Config.NORMALIZE_MODE == "rms_global":
        data_3d = normalize_rms_global(data_3d)
    elif Config.NORMALIZE_MODE == "max_global":
        data_3d = normalize_max_global(data_3d)
    elif Config.NORMALIZE_MODE == "trace":
        print("\nTrace normalization: will be applied during sample generation")
    else:
        print("\nNo normalization: using raw amplitude data")
    
    # Step 5: Generate training samples
    print("\n" + "="*80)
    print("Step 3: Generating training samples")
    print("="*80)
    
    generate_training_samples(data_3d, time_samples_abs, inlines, xlines, well_labels_dict)
    
    # Step 6: Generate prediction samples
    print("\n" + "="*80)
    print("Step 4: Generating prediction samples")
    print("="*80)
    
    generate_prediction_samples(data_3d, time_samples_abs, inlines, xlines)
    
    # Summary
    print("\n" + "="*80)
    print("Preprocessing completed!")
    print("="*80)
    
    suffix = get_normalization_suffix()
    print(f"\nOutput files:")
    print(f"  Training samples: {Config.OUTPUT_PATH}/training_samples/")
    print(f"    - training_samples{suffix}.txt")
    print(f"    - training_info{suffix}.csv")
    print(f"  Prediction samples: {Config.OUTPUT_PATH}/prediction_samples/")
    print(f"    - prediction_samples{suffix}.txt")
    print(f"    - prediction_info{suffix}.csv")
    print(f"\nNormalization mode: {Config.NORMALIZE_MODE}")


if __name__ == "__main__":
    main()
