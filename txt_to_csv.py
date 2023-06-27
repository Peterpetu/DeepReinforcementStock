import pandas as pd

def txt_to_csv(txt_filename, csv_filename):
    # Load the data from the txt file
    data = pd.read_csv(txt_filename)

    # Save the data to a csv file
    data.to_csv(csv_filename, index=False)

    print(f"Data saved to {csv_filename}")

# Usage
txt_to_csv('ibm.us.txt', 'ibm.us.csv')
