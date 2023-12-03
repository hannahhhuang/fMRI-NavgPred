import os
import pandas as pd

def read_and_label_files(folder_path, subject_code):
    file_names = [f'processed_run00{i}.csv' for i in range(1, 7)]
    dfs = []

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Subject'] = subject_code  # Add subject code as a column
            df['Run'] = file_name[:-4]  # Add run identifier as a column
            dfs.append(df)

    return pd.concat(dfs)

all_data_df = pd.DataFrame()

base_dir = '/Users/xingliu/Documents/processedOutput'  # Replace with your base directory path

for subject_num in range(2, 23):
    if subject_num == 11:  # Skip subject 11
        continue

    subject_code = f's{str(subject_num).zfill(2)}'
    subject_folder = os.path.join(base_dir, subject_code)

    if os.path.exists(subject_folder):
        subject_df = read_and_label_files(subject_folder, subject_code)
        all_data_df = pd.concat([all_data_df, subject_df], ignore_index=True)

output_file = os.path.join(base_dir, 'all_subjects_stacked.csv')
all_data_df.to_csv(output_file, index=False)
print('Processed and saved all subjects data')
