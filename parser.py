import logging
import os
import json
import re
import signal
import subprocess
import pandas as pd
from pyverilog.vparser.parser import parse
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pathlib import Path

logging.basicConfig(level=logging.INFO)

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Verilog parsing took too long")

def get_top_module_name(verilog_code):
    """Extract the top module name from Verilog code"""
    match = re.search(r'^\s*module\s+(\w+)', verilog_code, re.MULTILINE)
    return match.group(1) if match else None

def extract_circuit_features(verilog_code, filename='test.v'):
    """Extract circuit features from Verilog code using PyVerilog and Yosys."""
    filepath = Path(filename)

    # Write the code to a temporary file
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(verilog_code.lstrip())

    if filepath.stat().st_size == 0:
        print("Error: Verilog file is empty after writing.")
        return None

    top = get_top_module_name(verilog_code)
    features = {'module_name': top}
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)  # Set a 10-second timeout
    # Attempt PyVerilog analysis
    try:
        ast, directives = parse([filename])
        analyzer = VerilogDataflowAnalyzer([filename], top)
        analyzer.generate()

        # Get PyVerilog extracted features
        features.update({
            'signal_count': len(analyzer.getSignals()),
            'term_count': len(analyzer.getTerms()),
            'instance_count': len(analyzer.getInstances()),
            'bind_count': len(analyzer.getBinddict()),
            'estimated_depth': estimate_depth_from_ast(ast),
            'module_type': classify_module_type(verilog_code, ast),
        })
        features.update(count_operators(verilog_code))
        features.update(analyze_control_structures(ast))

    except TimeoutException:
        logging.error(f"Skipping: Verilog parsing took too long for {filename}")
        return None
    except Exception as e:
        logging.error(f"Skipping: Error analyzing {filename} - {str(e)}")
        return None

    # Run Yosys analysis
    try:
        yosys_cmd = f"yosys -p 'read_verilog {filename}; proc; opt; stat'"
        yosys_process = subprocess.Popen(yosys_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        yosys_stdout, yosys_stderr = yosys_process.communicate()

        if yosys_process.returncode != 0:
            print(f"Yosys error: {yosys_stderr.decode('utf-8')}")
        else:
            yosys_output = yosys_stdout.decode('utf-8')

            def parse_yosys_stats(yosys_output):
                stats = {}
                lines = yosys_output.splitlines()

                # Extract number of wires, cells, etc.
                for line in lines:
                    parts = line.split()
                    if len(parts) < 2:
                        continue  # Skip invalid lines

                    try:
                        if 'Number of wires' in line:
                            stats['num_wires'] = int(parts[-1])  # Extract last number
                        elif 'Number of cells' in line:
                            stats['num_cells'] = int(parts[-1])  

                        # Extract individual cell types (like $dff, $mux, etc.)
                        elif parts[0] in ['$dff', '$mux', '$not', '$and', '$or', '$xor']:
                            stats[f"{parts[0][1:]}_count"] = int(parts[1])  # Convert second element to int
                    except ValueError:
                        continue  # Skip if conversion fails

                return stats

            features.update(parse_yosys_stats(yosys_output))
    except TimeoutException:
        logging.error(f"Skipping: Verilog parsing took too long for {filename}")
        return None
    except Exception as e:
        logging.error(f"Skipping: Error analyzing {filename} - {str(e)}")
        return None
    signal.alarm(0)
    return features
    
    
    

def count_operators(verilog_code):
    """Count logic operators in the Verilog code"""
    return {
        'and_count': verilog_code.count('&') + verilog_code.count('&&'),
        'or_count': verilog_code.count('|') + verilog_code.count('||'),
        'not_count': verilog_code.count('!'),
        'xor_count': verilog_code.count('^'),
        'plus_count': verilog_code.count('+')
    }

def estimate_depth_from_ast(ast):
    """Estimate circuit depth from the AST when critical path analysis isn't available"""
    
    def traverse_node(node, current_depth=0):
        max_depth = current_depth
        
        # Check if this is a leaf node
        if not hasattr(node, 'children'):
            return current_depth
        
        # If not a leaf, traverse children
        for child in node.children():
            if child is not None:
                child_depth = traverse_node(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    return traverse_node(ast)

def analyze_control_structures(ast):
    """Count control structures in the AST"""
    return {
        'if_count': 10,  # Placeholder - real implementation would count from AST
        'case_count': 5,  # Placeholder
        'always_blocks': 3,  # Placeholder
    }

def classify_module_type(verilog_code, ast):
    """Classify the module type based on code patterns"""
    code_lower = verilog_code.lower()
    if 'always @(posedge clk' in code_lower:
        if 'memory' in code_lower or 'ram' in code_lower or 'rom' in code_lower:
            return 'Memory'
        elif 'count' in code_lower:
            return 'Counter'
        else:
            return 'Sequential'
    elif 'tx' in code_lower or 'rx' in code_lower or 'uart' in code_lower:
        return 'Interface'
    elif '+' in code_lower and ('*' in code_lower or '/' in code_lower):
        return 'Arithmetic'
    elif 'grant' in code_lower or 'arbiter' in code_lower or 'priority' in code_lower:
        return 'Control'
    else:
        return 'Combinational'

def process_verilog_dataset(data):
    """Process a dataset of Verilog code examples"""
    results = []
    
    total = len(data)
    for i, row in enumerate(data):
        if i % 100 == 0:
            print(f"Processing {i}/{total}...")
            
        instruction = row.get('Instruction', '')
        response = row.get('Response', [''])[0]  # Assuming the first element contains the Verilog code
        
        # Extract features
        features = extract_circuit_features(response)
        
        results.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

def main():
    # Load your dataset
    with open('verilog_dataset.json', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Process the dataset
    result_df = process_verilog_dataset(data[:10000])
    
    # Save to CSV
    result_df.to_csv('verilog_circuit_features.csv', index=False)
    
    # Print summary
    print("Dataset creation complete!")
    print(f"Total entries: {len(result_df)}")
    print("\nSample entries:")
    print(result_df.head())
    
    print("\nFeature statistics:")
    print(result_df.describe())

if __name__ == "__main__":
    main()
