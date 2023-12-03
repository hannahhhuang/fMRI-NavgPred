import pandas as pd
import numpy as np
import os

class OrientationDataHandler:
    def __init__(self, file_path):
        # Read the data from the file
        self.data = pd.read_csv(file_path, delimiter='\t', on_bad_lines='skip')

    def _find_closest_index(self, target_time):
        # Find the index of the closest time
        absolute_differences = np.abs(self.data['Time'] - target_time)
        closest_index = absolute_differences.idxmin()
        return closest_index

    def get_average_orientations(self, start_time, interval):
        max_time = self.data['Time'].max()
        current_time = start_time
        times = []
        average_orientations = []

        while current_time <= max_time:
            times.append(current_time)
            closest_index = self._find_closest_index(current_time)

            # Find the orientations around the closest time point
            orientations = [self.data.at[closest_index, 'Orientation']]
            if closest_index > 0:
                orientations.append(self.data.at[closest_index - 1, 'Orientation'])
            if closest_index + 1 < len(self.data):
                orientations.append(self.data.at[closest_index + 1, 'Orientation'])

            # Calculate the average orientation
            average_orientation = np.mean(orientations)
            average_orientations.append(average_orientation)

            # Move to the next interval
            current_time += interval

        return times, average_orientations

    def write_orientations_to_file(self, output_file, start_time, interval):
        # Get the average orientations
        times, averages = self.get_average_orientations(start_time, interval)

        data = {'Time': times, 'AverageOrientation': averages}

        averages_df = pd.DataFrame(data)

        averages_df.to_csv(output_file, index=False)


# data_dir = '/Users/xingliu/Documents/processedNavigationData/s04/BehavioralData_s04'
#
# output_dir = '/Users/xingliu/Documents/processedOutput/S04'
#
# # Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Create the file paths
# file_paths = [os.path.join(data_dir, f's04_indoor_noBarrier_run{str(i).zfill(3)}.tsv') for i in range(7, 13)]
#
# start_time = 0
# interval = 1.5
#
# # Process each file
# for i in range(1, 7):
#     file_path = file_paths[i - 1]
#     processor = OrientationDataHandler(file_path)
#
#     output_filename = os.path.join(output_dir, f'processed_run00{i}.csv')
#
#     processor.write_orientations_to_file(output_filename, start_time, interval)

# Function to process a single subject
# Function to process a single subject
def process_subject(subject_number):
    data_dir = f'/Users/xingliu/Documents/processedNavigationData/{subject_number}/BehavioralData_{subject_number}'
    output_dir = f'/Users/xingliu/Documents/processedOutput/{subject_number}'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = 0
    interval = 1.5
    output_file_counter = 1
    processed_files = set()
    # Iterate over possible run numbers
    for run_num in range(1, 13):
        for environment in ['indoor', 'outdoor']:
            # Standardize barrier naming to lowercase
            file_name = f'{subject_number}_{environment}_nobarrier_run{str(run_num).zfill(3)}.tsv'.lower()
            file_path = os.path.join(data_dir, file_name)

            # Check and process if not already done
            if os.path.exists(file_path.lower()) and file_name not in processed_files:
                print(f"Processing: {file_name}")
                processor = OrientationDataHandler(file_path)
                output_filename = os.path.join(output_dir, f'processed_run{str(output_file_counter).zfill(3)}.csv')
                processor.write_orientations_to_file(output_filename, start_time, interval)
                output_file_counter += 1
                processed_files.add(file_name)  # Mark as processed
            else:
                print(f"File not found or already processed: {file_path}")
# Process all subjects from S05 to S22
for subject_num in range(5, 23):
    subject_code = f's{str(subject_num).zfill(2)}'
    process_subject(subject_code)






