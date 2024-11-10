import glob
import pandas as pd

# Define the parameters
eyes_options = ['left', 'right']
variant_options = [18, 50]
participant_ids = range(1, 52)

# Dictionary to store data
data = []

# Loop over all combinations of EYES, VARIANT, and PARTICIPANT_ID
for eyes in eyes_options:
    for variant in variant_options:
        for participant_id in participant_ids:
            # Construct the file path
            file_path = f"/netscratch/shah/pupil-size-estimation-with-super-resolution/local/local_ResNet{variant}_{eyes}_eyes/mlruns/0/{participant_id}/metrics/test_epoch_running_loss"
            
            # Try to open and read the file
            try:
                with open(file_path, 'r') as file:
                    # Extract the second column from each line
                    for line in file:
                        columns = line.strip().split()
                        if len(columns) > 1:
                            # Append the extracted data to the list
                            data.append({
                                'Eyes': eyes,
                                'Variant': variant,
                                'Participant_ID': participant_id,
                                'MAE': float(columns[1])  # Convert to float if needed
                            })
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

# Convert the data into a DataFrame for analysis
df = pd.DataFrame(data)
df.to_csv("./local/extracted_data.csv", index=False)

# Group by 'Eyes' and 'Variant', and calculate mean and std for 'MAE'
summary_df = df.groupby(['Variant', 'Eyes'])['MAE'].agg(['mean', 'std']).reset_index()

# Rename columns for clarity
summary_df.columns = ['Variant', 'Eyes', 'Average', 'Standard_Deviation']

# Add a new column to combine 'Variant' and 'Eyes' for a more descriptive label
summary_df['Label'] = 'ResNet' + summary_df['Variant'].astype(str) + '_' + summary_df['Eyes'] + '_eyes'

# Set 'Label' as the index and only keep relevant columns
summary_df = summary_df.set_index('Label')[['Average', 'Standard_Deviation']]

# Display the final table
print(summary_df)
summary_df.to_csv("./local/summary_statistics.csv")