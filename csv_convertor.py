import csv
import ast

def convert_txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    data_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):  # Remove surrounding quotes
            line = line[1:-1]
        try:
            parsed_dict = ast.literal_eval(line)  # Convert string to dictionary
            data_list.append(parsed_dict)
        except (SyntaxError, ValueError) as e:
            print(f"Skipping invalid line: {line}\nError: {e}")

    if not data_list:
        print("No valid data found.")
        return

    # Extract headers from the first dictionary
    headers = set()
    for entry in data_list:
        headers.update(entry.keys())
    headers = sorted(headers)  # Sort headers for consistency

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_list)

    print(f"CSV file '{output_file}' created successfully.")

# Usage
input_file = "verilog_circuit_features.txt"  # Update with your actual filename
output_file = "modules.csv"
convert_txt_to_csv(input_file, output_file)
