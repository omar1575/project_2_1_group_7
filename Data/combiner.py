import os
import pandas as pd


output_file = "Data/Data.csv"
root_dir = "Data/."


if __name__ == '__main__':

    all_csv_files = []
    output_path = os.path.abspath(output_file)

    for path, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".csv"):
                full_path = os.path.join(path, filename)

                if os.path.abspath(output_path) == os.path.abspath(full_path):
                    #Output file
                    continue
                all_csv_files.append(full_path)


    for f in all_csv_files:
        print(" -", f)

    dfs = []
    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Combined CSV saved as: {output_file}")

