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


data_dir = '/Users/xingliu/Documents/processedNavigationData/s02/BehavioralData_s02'

output_dir = '/Users/xingliu/Documents/processedOutput'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the file paths
file_paths = [os.path.join(data_dir, f's02_outdoor_noBarrier_run00{i}.tsv') for i in range(1, 7)]

start_time = 0
interval = 1.5

# Process each file
for i in range(1, 7):
    file_path = file_paths[i - 1]
    processor = OrientationDataHandler(file_path)

    output_filename = os.path.join(output_dir, f'processed_run00{i}.csv')

    processor.write_orientations_to_file(output_filename, start_time, interval)





