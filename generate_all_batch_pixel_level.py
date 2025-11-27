"""
Generate all_batch.csv with pixel-level preprocessed data
Applies batch-specific preprocessing logic from individual batch code folders
WITHOUT averaging - outputs pixel-level data so that when aggregated,
the device-level values match Batch_Analysis/batch XX/results.csv files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PixelLevelBatchProcessor:
    def __init__(self, synology_base, workspace_base):
        self.synology_base = Path(synology_base)
        self.workspace_base = Path(workspace_base)
        self.parameters = ['PCE', 'Max_Power', 'FF', 'J_sc', 'V_oc', 'HI', 'R_shunt', 'R_series']
        
        # Load runs.csv for station and stack mapping
        self.device_stack_map = {}
        self.device_station_map = {}
        self._load_runs_csv()
        
    def find_largest_lifetime_files(self, date_folder):
        """Find the largest (most recent) lifetime CSV files in a date folder, prioritizing _new.csv files"""
        # First try to find *_new.csv files (corrected files)
        new_lifetime_files = list(date_folder.glob("*lifetime_new.csv"))
        
        if new_lifetime_files:
            print(f"    Found {len(new_lifetime_files)} corrected (_new.csv) lifetime files")
            lifetime_files = new_lifetime_files
        else:
            # Fallback to original files if no _new files exist
            lifetime_files = list(date_folder.glob("*lifetime.csv"))
            # Filter out _new files in case glob picked them up
            lifetime_files = [f for f in lifetime_files if not f.stem.endswith('_new')]
            print(f"    Found {len(lifetime_files)} original lifetime files")
        
        if not lifetime_files:
            return []
        
        # Group by pixel (a, b, c, d)
        pixel_files = {}
        for file in lifetime_files:
            # Extract pixel from filename (last part before 'lifetime.csv' or 'lifetime_new.csv')
            parts = file.stem.split()
            if file.stem.endswith('_new'):
                pixel = parts[-2]  # For *_new.csv files, pixel is second from last
            else:
                pixel = parts[-2]  # For original files
            
            if pixel not in pixel_files:
                pixel_files[pixel] = []
            pixel_files[pixel].append(file)
        
        # For each pixel, find the largest file (or just take the _new file if available)
        largest_files = []
        for pixel, files in pixel_files.items():
            if any(f.stem.endswith('_new') for f in files):
                # Prioritize _new files
                new_files = [f for f in files if f.stem.endswith('_new')]
                largest_file = max(new_files, key=lambda f: f.stat().st_size)
            else:
                largest_file = max(files, key=lambda f: f.stat().st_size)
            largest_files.append(largest_file)
        
        return largest_files
    
    def _load_runs_csv(self):
        """Load device-stack-station mapping from runs.csv"""
        try:
            runs_file = self.workspace_base / 'runs.csv'
            if not runs_file.exists():
                print(f"Warning: runs.csv not found at {runs_file}")
                return
            
            runs_df = pd.read_csv(runs_file)
            
            # Parse LS section (columns 0-4)
            ls_section = runs_df.iloc[:, 0:5]
            ls_section.columns = ['Batch_ID', 'Sheet_ID', 'Device_ID', 'Station', 'Stack']
            ls_clean = ls_section[ls_section['Batch_ID'].notna()].copy()
            
            for _, row in ls_clean.iterrows():
                batch = int(row['Batch_ID'])
                device_full = str(row['Device_ID'])
                station = str(row['Station']).strip()
                stack = str(row['Stack']).strip()
                
                parts = device_full.split('-')
                if len(parts) >= 4:
                    device_short = f"{parts[2]}-{parts[3]}"
                    key = (batch, device_short)
                    self.device_stack_map[key] = stack
                    self.device_station_map[key] = station
            
            # Parse Sunbrick section (columns 6-10)
            sunbrick_section = runs_df.iloc[:, 6:11]
            sunbrick_section.columns = ['Batch_ID', 'Sheet_ID', 'Device_ID', 'Station', 'Stack']
            sunbrick_clean = sunbrick_section[sunbrick_section['Batch_ID'].notna()].copy()
            
            for _, row in sunbrick_clean.iterrows():
                batch = int(row['Batch_ID'])
                device_full = str(row['Device_ID'])
                station = str(row['Station']).strip()
                stack = str(row['Stack']).strip()
                
                parts = device_full.split('-')
                if len(parts) >= 4:
                    device_short = f"{parts[2]}-{parts[3]}"
                    key = (batch, device_short)
                    self.device_stack_map[key] = stack
                    self.device_station_map[key] = station
            
            # Parse EC section (columns 12-16)
            ec_section = runs_df.iloc[:, 12:17]
            ec_section.columns = ['Batch_ID', 'Sheet_ID', 'Device_ID', 'Station', 'Stack']
            ec_clean = ec_section[ec_section['Batch_ID'].notna()].copy()
            
            for _, row in ec_clean.iterrows():
                batch = int(row['Batch_ID'])
                device_full = str(row['Device_ID'])
                station = str(row['Station']).strip()
                stack = str(row['Stack']).strip()
                
                parts = device_full.split('-')
                if len(parts) >= 4:
                    device_short = f"{parts[2]}-{parts[3]}"
                    key = (batch, device_short)
                    self.device_stack_map[key] = stack
                    self.device_station_map[key] = station
            
            print(f"Loaded {len(self.device_stack_map)} device-stack mappings from runs.csv")
            
        except Exception as e:
            print(f"Error loading runs.csv: {e}")
    
    def analyze_reference_pattern(self, results_file):
        """Analyze results.csv to determine if F or R values are typically higher"""
        try:
            df = pd.read_csv(results_file)
            
            # Find rows where Hold Voltage contains 'Max Power'
            max_power_rows = df[df['Hold Voltage'].str.contains('Max Power', na=False)]
            
            if len(max_power_rows) == 0:
                return None
            
            # Take next 8 rows including the first match
            start_idx = max_power_rows.index[0]
            reference_rows = df.iloc[start_idx:start_idx+8]
            
            # Count F vs R for PCE values
            f_higher_count = 0
            total_comparisons = 0
            
            for idx in range(0, len(reference_rows)-1, 2):
                if idx+1 < len(reference_rows):
                    row_f = reference_rows.iloc[idx]
                    row_r = reference_rows.iloc[idx+1]
                    
                    if (row_f['Scan Direction'] == 'F' and row_r['Scan Direction'] == 'R'):
                        pce_f = row_f['PCE (%)']
                        pce_r = row_r['PCE (%)']
                        
                        if pd.notna(pce_f) and pd.notna(pce_r):
                            if pce_f > pce_r:
                                f_higher_count += 1
                            total_comparisons += 1
            
            if total_comparisons > 0:
                return f_higher_count / total_comparisons > 0.5  # True if F typically higher
            
        except Exception as e:
            print(f"      Error analyzing reference pattern: {e}")
        
        return None
    
    def swap_all_parameters(self, df, row_index):
        """Swap F and R values for all parameters in a given row"""
        # Find all columns that contain '_F' and have corresponding '_R' columns
        for col in df.columns:
            if '_F ' in col:
                # Generate the corresponding R column name
                r_col = col.replace('_F ', '_R ')
                
                if r_col in df.columns:
                    # Swap the values
                    temp = df.at[row_index, col]
                    df.at[row_index, col] = df.at[row_index, r_col]
                    df.at[row_index, r_col] = temp
    
    def detect_swaps(self, df, pixel, device_name, f_typically_higher):
        """Detect and correct swapped values in the lifetime data"""
        corrections_made = []
        df_corrected = df.copy()
        
        # Convert time column to numeric for trend analysis
        time_col = 'Time (hrs)'
        if time_col not in df.columns:
            return df_corrected, corrections_made
        
        # Use PCE as the reference parameter to detect swaps
        param = 'PCE'
        f_col = f'{param}_F (%)'
        r_col = f'{param}_R (%)'
        
        if f_col not in df_corrected.columns or r_col not in df_corrected.columns:
            return df_corrected, corrections_made
        
        # Get clean data (remove NaN values)
        mask = pd.notna(df_corrected[f_col]) & pd.notna(df_corrected[r_col])
        if mask.sum() < 3:  # Need at least 3 points for trend analysis
            return df_corrected, corrections_made
        
        # Step 1: Fix initial values based on reference pattern
        if len(df_corrected) > 0:
            first_f = df_corrected.iloc[0][f_col]
            first_r = df_corrected.iloc[0][r_col]
            
            if pd.notna(first_f) and pd.notna(first_r):
                # Check if initial values match expected pattern
                initial_swap_needed = False
                if f_typically_higher and first_f < first_r:
                    initial_swap_needed = True
                elif not f_typically_higher and first_f > first_r:
                    initial_swap_needed = True
                
                if initial_swap_needed:
                    # Swap all parameters for the first row
                    self.swap_all_parameters(df_corrected, 0)
                    corrections_made.append(f"Row 0: Swapped ALL parameters (initial correction)")
        
        # Step 2: Progressive correction based on PCE smoothness
        for i in range(len(df_corrected)):
            if not mask.iloc[i]:
                continue
                
            curr_f = df_corrected.iloc[i][f_col]
            curr_r = df_corrected.iloc[i][r_col]
            
            # Determine expected relationship based on reference pattern
            expected_f_higher = f_typically_higher
            
            # If we're past the first few points, also consider the local trend
            if i >= 2:
                # Look at the last 2 valid points to establish local pattern
                recent_points = []
                for j in range(max(0, i-3), i):
                    if mask.iloc[j]:
                        recent_f = df_corrected.iloc[j][f_col]
                        recent_r = df_corrected.iloc[j][r_col]
                        recent_points.append((recent_f, recent_r))
                
                if len(recent_points) >= 1:
                    # Count how many recent points have F > R
                    f_higher_count = sum(1 for f, r in recent_points if f > r)
                    local_f_higher = f_higher_count > len(recent_points) / 2
                    
                    # Use local pattern if it's consistent, otherwise use reference
                    expected_f_higher = local_f_higher
            
            # Check if current values violate the expected pattern
            values_swapped = False
            if expected_f_higher and curr_f < curr_r - 0.3:  # F should be higher
                values_swapped = True
            elif not expected_f_higher and curr_r < curr_f - 0.3:  # R should be higher
                values_swapped = True
            
            # Additional check: detect sudden crossing that breaks smoothness
            if i > 0 and mask.iloc[i-1]:
                prev_f = df_corrected.iloc[i-1][f_col]
                prev_r = df_corrected.iloc[i-1][r_col]
                
                # Calculate changes
                f_change = curr_f - prev_f
                r_change = curr_r - prev_r
                
                # Detect opposite movements (strong indicator of swap)
                if abs(f_change) > 0.5 and abs(r_change) > 0.5:
                    if (f_change > 0 and r_change < 0) or (f_change < 0 and r_change > 0):
                        values_swapped = True
                
                # Detect sudden relationship reversal
                prev_f_higher = prev_f > prev_r
                curr_f_higher = curr_f > curr_r
                if prev_f_higher != curr_f_higher and abs(curr_f - curr_r) > 1.0:
                    # Relationship suddenly reversed with significant gap
                    values_swapped = True
            
            # Perform swap for ALL parameters if needed
            if values_swapped:
                self.swap_all_parameters(df_corrected, i)
                corrections_made.append(f"Row {i}: Swapped ALL parameters")
        
        return df_corrected, corrections_made
    
    def process_pixel_data(self, df, pixel, f_typically_higher, device_name, batch_num, device_id, sheet_id, station, stack, date_str=""):
        """Process pixel data (from file or combined data) and return pixel-level data"""
        pixel_data = []
        pixel_id = pixel  # Keep pixel identifier (a, b, c, d)
        
        try:
            # Detect and correct swaps
            df_corrected, corrections = self.detect_swaps(df, pixel, device_name, f_typically_higher)
            
            if corrections:
                print(f"      Pixel {pixel}: Made {len(corrections)} swap corrections")
            
            # Round timestamps to nearest 0.1 hour for alignment
            df_corrected['Time_hrs_rounded'] = df_corrected['Time (hrs)'].round(1)
            
            # Define the actual column mappings
            column_mappings = {
                'PCE': ('PCE_F (%)', 'PCE_R (%)'),
                'FF': ('FF_F (%)', 'FF_R (%)'),
                'J_sc': ('J_sc_F (mA/cm2)', 'J_sc_R (mA/cm2)'),
                'V_oc': ('V_oc_F (V)', 'V_oc_R (V)'),
                'Max_Power': ('Max Power_F (mW/cm2)', 'Max Power_R (mW/cm2)'),
                'R_shunt': ('R_shunt_F (Ohm.cm2)', 'R_shunt_R (Ohm.cm2)'),
                'R_series': ('R_series_F (Ohm.cm2)', 'R_series_R (Ohm.cm2)')
            }
            
            # HI is a single value (same for F and R)
            hi_col = 'HI (%)' if 'HI (%)' in df_corrected.columns else None
            
            # Process each timestamp and create pixel-level rows
            for idx, row in df_corrected.iterrows():
                time_hrs = row['Time_hrs_rounded']
                
                # Get HI value
                hi_value = row[hi_col] if hi_col and pd.notna(row[hi_col]) else np.nan
                
                # Extract F and R values for each parameter
                f_values = {}
                r_values = {}
                
                for param, (f_col, r_col) in column_mappings.items():
                    if f_col in row and r_col in row:
                        f_values[param] = row[f_col] if pd.notna(row[f_col]) else np.nan
                        r_values[param] = row[r_col] if pd.notna(row[r_col]) else np.nan
                
                # Create pixel-level row
                if any(pd.notna(v) for v in f_values.values()):
                    pixel_row = {
                        'Batch': batch_num,
                        'Device_ID': device_id,
                        'Sheet_ID': sheet_id,
                        'Pixel_ID': f"{sheet_id}-{pixel_id}",
                        'Station': station,
                        'Stack': stack,
                        'Date': date_str,
                        'Time_hrs': time_hrs,
                        'PCE_F': f_values.get('PCE', np.nan),
                        'PCE_R': r_values.get('PCE', np.nan),
                        'FF_F': f_values.get('FF', np.nan),
                        'FF_R': r_values.get('FF', np.nan),
                        'J_sc_F': f_values.get('J_sc', np.nan),
                        'J_sc_R': r_values.get('J_sc', np.nan),
                        'V_oc_F': f_values.get('V_oc', np.nan),
                        'V_oc_R': r_values.get('V_oc', np.nan),
                        'Max_Power_F': f_values.get('Max_Power', np.nan),
                        'Max_Power_R': r_values.get('Max_Power', np.nan),
                        'R_shunt_F': f_values.get('R_shunt', np.nan),
                        'R_shunt_R': r_values.get('R_shunt', np.nan),
                        'R_series_F': f_values.get('R_series', np.nan),
                        'R_series_R': r_values.get('R_series', np.nan),
                        'HI_F': hi_value,
                        'HI_R': hi_value
                    }
                    pixel_data.append(pixel_row)
            
            print(f"      Pixel {pixel}: Processed {len(pixel_data)} data points")
            
        except Exception as e:
            print(f"      Error processing pixel {pixel}: {e}")
        
        return pixel_data
    
    def process_pixel_file(self, lifetime_file, f_typically_higher, device_name, batch_num, device_id, sheet_id, station, stack):
        """Process a single lifetime CSV file and return pixel-level data"""
        try:
            # Read the CSV file
            df = pd.read_csv(lifetime_file)
            
            # Extract pixel from filename
            pixel = lifetime_file.stem.split()[-2] if not lifetime_file.stem.endswith('_new') else lifetime_file.stem.split()[-3]
            
            # Extract date from the file path (date folder name)
            date_str = lifetime_file.parent.name if len(lifetime_file.parent.name) == 10 else ""
            
            return self.process_pixel_data(df, pixel, f_typically_higher, device_name, batch_num, device_id, sheet_id, station, stack, date_str)
            
        except Exception as e:
            print(f"      Error reading {lifetime_file.name}: {e}")
            return []
    
    def combine_batch_65_dates(self, device_folder, pixel):
        """Combine lifetime files from 2025-10-15 and 2025-10-17 for batch 65"""
        date_folders = []
        for folder in device_folder.iterdir():
            if folder.is_dir() and folder.name in ['2025-10-15', '2025-10-17']:
                date_folders.append(folder)
        
        if len(date_folders) != 2:
            return None
        
        date_folders.sort()  # 2025-10-15 first, then 2025-10-17
        
        combined_data = []
        max_time_hrs = 0
        
        # Process first date (2025-10-15)
        first_files = self.find_largest_lifetime_files(date_folders[0])
        if first_files:
            for file in first_files:
                file_pixel = file.stem.split()[-2] if not file.stem.endswith('_new') else file.stem.split()[-3]
                if file_pixel == pixel:
                    df = pd.read_csv(file)
                    combined_data.append(df)
                    if len(df) > 0 and 'Time (hrs)' in df.columns:
                        max_time_hrs = df['Time (hrs)'].max()
                    break
        
        # Process second date (2025-10-17) with time offset
        second_files = self.find_largest_lifetime_files(date_folders[1])
        if second_files:
            for file in second_files:
                file_pixel = file.stem.split()[-2] if not file.stem.endswith('_new') else file.stem.split()[-3]
                if file_pixel == pixel:
                    df = pd.read_csv(file)
                    # Adjust time - continue from where first file ended
                    if 'Time (hrs)' in df.columns and len(df) > 0:
                        time_offset = max_time_hrs + 1.0
                        df_adjusted = df.copy()
                        df_adjusted['Time (hrs)'] = df['Time (hrs)'] + time_offset
                        combined_data.append(df_adjusted)
                    break
        
        # Combine the data
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        
        return None
    
    def get_device_metadata(self, device_name, batch_num):
        """Extract metadata from device name and batch info"""
        # Example: "#B25-53-S001-A4" -> Batch: 53, Device_ID: S001-A4, Sheet_ID: S001
        parts = device_name.split('-')
        if len(parts) >= 4:
            substrate = parts[2]  # Gets "S001", "S002", etc.
            pixel_group = parts[3]  # Gets "A4", "B3", etc.
            device_id = f"{substrate}-{pixel_group}"
            sheet_id = substrate
        else:
            device_id = device_name
            sheet_id = device_name
        
        # Get Station and Stack from runs.csv mapping
        key = (int(batch_num), device_id)
        station = self.device_station_map.get(key, "Sunbrick")  # Default to Sunbrick if not found
        stack = self.device_stack_map.get(key, "Unknown")  # Default to Unknown if not found
        
        return device_id, sheet_id, station, stack
    
    def process_batch(self, batch_num, data_source_path):
        """Process a single batch and return pixel-level data"""
        print(f"\n=== Processing Batch {batch_num} ===")
        print(f"Data source: {data_source_path}")
        
        all_pixel_data = []
        
        # Find all device folders
        device_folders = []
        for folder in data_source_path.iterdir():
            if folder.is_dir() and folder.name.startswith('#B25-'):
                device_folders.append(folder)
        
        # For batch 65, also check archive subfolder
        if batch_num == '65':
            archive_folder = data_source_path / 'archive'
            if archive_folder.exists():
                print(f"  Also checking archive subfolder...")
                for folder in archive_folder.iterdir():
                    if folder.is_dir() and folder.name.startswith('#B25-'):
                        device_folders.append(folder)
        
        if not device_folders:
            print(f"  No device folders found for batch {batch_num}")
            return all_pixel_data
        
        print(f"Found {len(device_folders)} devices")
        
        for device_folder in sorted(device_folders):
            device_name = device_folder.name
            device_id, sheet_id, station, stack = self.get_device_metadata(device_name, batch_num)
            
            print(f"  Processing device: {device_name} ({device_id})")
            
            # Get oldest date folder
            date_folders = []
            for folder in device_folder.iterdir():
                if folder.is_dir() and folder.name.startswith('2025-'):
                    date_folders.append(folder)
            
            if not date_folders:
                print(f"    No date folders found")
                continue
            
            oldest_date_folder = min(date_folders)
            print(f"    Using oldest date: {oldest_date_folder.name}")
            
            # Find results.csv file for reference pattern analysis
            results_file = oldest_date_folder / f"{oldest_date_folder.name} {device_name} results.csv"
            f_typically_higher = None
            
            if results_file.exists():
                f_typically_higher = self.analyze_reference_pattern(results_file)
                if f_typically_higher is not None:
                    pattern_str = "F typically higher" if f_typically_higher else "R typically higher"
                    print(f"    Reference pattern: {pattern_str}")
            
            if f_typically_higher is None:
                print(f"    Using default: F typically higher")
                f_typically_higher = True
            
            # Special handling for batch 65 - combine data from multiple dates
            if batch_num == '65':
                print(f"    Special batch 65 processing: combining dates 2025-10-15 and 2025-10-17")
                
                # Get all pixels that might exist
                all_pixels = set()
                for date_folder in date_folders:
                    files = self.find_largest_lifetime_files(date_folder)
                    for file in files:
                        pixel = file.stem.split()[-2] if not file.stem.endswith('_new') else file.stem.split()[-3]
                        all_pixels.add(pixel)
                
                # Process each pixel with combined data
                for pixel in sorted(all_pixels):
                    combined_df = self.combine_batch_65_dates(device_folder, pixel)
                    if combined_df is not None:
                        pixel_data = self.process_pixel_data(
                            combined_df, pixel, f_typically_higher, device_name,
                            batch_num, device_id, sheet_id, station, stack, "2025-10-15"
                        )
                        all_pixel_data.extend(pixel_data)
            else:
                # Find largest lifetime files (prioritizes _new.csv)
                lifetime_files = self.find_largest_lifetime_files(oldest_date_folder)
                
                if not lifetime_files:
                    print(f"    No lifetime files found")
                    continue
                
                print(f"    Processing {len(lifetime_files)} pixel files")
                
                # Process each pixel file
                for lifetime_file in lifetime_files:
                    pixel_data = self.process_pixel_file(
                        lifetime_file, f_typically_higher, device_name,
                        batch_num, device_id, sheet_id, station, stack
                    )
                    all_pixel_data.extend(pixel_data)
        
        print(f"Batch {batch_num}: Generated {len(all_pixel_data)} pixel-level data points")
        return all_pixel_data
    
    def generate_all_batch_csv(self):
        """Generate all_batch.csv with pixel-level preprocessed data from all batches"""
        print("=" * 80)
        print("GENERATING ALL_BATCH.CSV WITH PIXEL-LEVEL PREPROCESSED DATA")
        print("=" * 80)
        
        # Define batch configurations
        batches = {
            '45': 'batch 45',
            '53': 'batch 53',
            '58': 'batch 58',
            '59': 'batch 59 + 53',  # Special combined folder
            '65': 'batch 65',
            '68': 'batch 68',
            '69': 'batch 69',
            '72': 'batch 72'
        }
        
        all_data = []
        
        for batch_num, folder_name in batches.items():
            data_source_path = self.synology_base / folder_name
            
            # Check if path exists
            if not data_source_path.exists():
                print(f"\nWARNING: Batch {batch_num} path does not exist: {data_source_path}")
                print("Skipping this batch...")
                continue
            
            # Process batch and collect pixel-level data
            batch_data = self.process_batch(batch_num, data_source_path)
            all_data.extend(batch_data)
        
        # Create DataFrame
        print(f"\n{'=' * 80}")
        print(f"Creating final DataFrame with {len(all_data)} total pixel-level rows")
        df_all_batch = pd.DataFrame(all_data)
        
        # Sort by Batch, Device_ID, Pixel_ID, Time_hrs
        df_all_batch = df_all_batch.sort_values(['Batch', 'Device_ID', 'Pixel_ID', 'Time_hrs'])
        
        # Save to all_batch.csv (overwrite existing file)
        output_file = self.workspace_base / "all_batch_new.csv"
        df_all_batch.to_csv(output_file, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"✓ all_batch_new.csv generated successfully!")
        print(f"Output file: {output_file}")
        print(f"Total rows: {len(df_all_batch)}")
        print(f"Columns: {list(df_all_batch.columns)}")
        print(f"\nSample data:")
        print(df_all_batch.head(10))
        print(f"{'=' * 80}")
        
        return df_all_batch


def main():
    """Main execution function"""
    # Set paths
    synology_base = r"C:\Users\MahekKamani\SynologyDrive\Rayleigh\Lab Data\Sunbrick station\device data\4-stability - data\AF study\AF test"
    workspace_base = r"c:\Users\MahekKamani\OneDrive - Rayleigh Solar Tech Inc\Desktop\AF_analysis"
    
    # Create processor
    processor = PixelLevelBatchProcessor(synology_base, workspace_base)
    
    # Generate all_batch.csv
    df = processor.generate_all_batch_csv()
    
    print("\n✓ Processing complete!")
    print("\nNext steps:")
    print("1. Use the new all_batch.csv to create device-level aggregation")
    print("2. Compare device-level values with Batch_Analysis/batch XX/results.csv files")
    print("3. Verify that values match exactly")


if __name__ == "__main__":
    main()
