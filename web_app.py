import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import os
import tempfile
import subprocess
from pathlib import Path
import sys
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Add title and description
st.set_page_config(page_title="Verilog Circuit Depth Predictor", layout="wide")
st.title("Verilog Circuit Depth Predictor")
st.markdown("""
This tool predicts the depth of a Verilog circuit before simulation using machine learning models.
Upload your Verilog file or paste your code to get an estimate of the circuit depth.
""")

# Load the models
@st.cache_resource
def load_models():
    models = {
        "LinearRegression": joblib.load("models/LinearRegression.pkl"),
        "RandomForest": joblib.load("models/RandomForest.pkl"),
        "GradientBoosting": joblib.load("models/GradientBoosting.pkl"),
        "XGBoost": joblib.load("models/XGBoost.pkl")
    }
    return models

try:
    models = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading models: {str(e)}")
    st.info("Make sure model files are in the 'models' directory")
    models = None

# Verilog feature extraction functions
def get_top_module_name(verilog_code):
    """Extract the top module name from Verilog code"""
    match = re.search(r'^\s*module\s+(\w+)', verilog_code, re.MULTILINE)
    return match.group(1) if match else None

def count_operators(verilog_code):
    """Count logic operators in the Verilog code"""
    return {
        'and_count': verilog_code.count('&') + verilog_code.count('&&'),
        'or_count': verilog_code.count('|') + verilog_code.count('||'),
        'not_count': verilog_code.count('!'),
        'xor_count': verilog_code.count('^'),
        'plus_count': verilog_code.count('+')
    }

def classify_module_type(verilog_code, ast=None):
    """Classify the module type based on code patterns"""
    code_lower = verilog_code.lower()
    
    # Create one-hot encoded features for module type
    module_types = {
        'module_type_Memory': 0,
        'module_type_Counter': 0,
        'module_type_Sequential': 0,
        'module_type_Interface': 0,
        'module_type_Arithmetic': 0,
        'module_type_Control': 0,
        'module_type_Combinational': 0
    }
    
    if 'always @(posedge clk' in code_lower:
        if 'memory' in code_lower or 'ram' in code_lower or 'rom' in code_lower:
            module_types['module_type_Memory'] = 1
        elif 'count' in code_lower:
            module_types['module_type_Counter'] = 1
        else:
            module_types['module_type_Sequential'] = 1
    elif 'tx' in code_lower or 'rx' in code_lower or 'uart' in code_lower:
        module_types['module_type_Interface'] = 1
    elif '+' in code_lower and ('*' in code_lower or '/' in code_lower):
        module_types['module_type_Arithmetic'] = 1
    elif 'grant' in code_lower or 'arbiter' in code_lower or 'priority' in code_lower:
        module_types['module_type_Control'] = 1
    else:
        module_types['module_type_Combinational'] = 1
        
    return module_types

def count_control_structures(verilog_code):
    """Count control structures in the Verilog code"""
    if_count = len(re.findall(r'if\s*\(', verilog_code))
    case_count = len(re.findall(r'case\s*\(', verilog_code))
    always_blocks = len(re.findall(r'always\s*@', verilog_code))
    
    return {
        'if_count': if_count,
        'case_count': case_count,
        'always_blocks': always_blocks
    }

def extract_basic_features(verilog_code):
    """Extract basic features without using external libraries"""
    features = {}
    
    # Count signals (wires, regs)
    features['signal_count'] = len(re.findall(r'\bwire\b|\breg\b', verilog_code))
    
    # Count instances (module instantiations)
    features['instance_count'] = len(re.findall(r'\b\w+\s+\w+\s*\(', verilog_code))
    
    # Count terms (assignments)
    features['term_count'] = len(re.findall(r'=', verilog_code))
    
    # Count bindings (connections)
    features['bind_count'] = len(re.findall(r'\.\w+\s*\(', verilog_code))
    
    # Get operators count
    features.update(count_operators(verilog_code))
    
    # Get control structures count
    features.update(count_control_structures(verilog_code))
    
    # Get module type
    features.update(classify_module_type(verilog_code))
    
    # Set default values for features that would normally come from Yosys
    features['num_wires'] = features['signal_count']
    features['num_cells'] = features['instance_count']
    features['dff_count'] = len(re.findall(r'always\s*@\s*\(\s*posedge', verilog_code))
    
    # Fixed: Count ternary operators for mux_count (? :)
    features['mux_count'] = len(re.findall(r'\?', verilog_code))
    
    # Re-use the values we already computed
    features['not_count'] = features['not_count']
    features['and_count'] = features['and_count']
    features['or_count'] = features['or_count']
    features['xor_count'] = features['xor_count']
    
    # Lines of code can be a proxy for complexity
    features['loc'] = len(verilog_code.splitlines())
    
    return features

def run_yosys_analysis(verilog_code):
    """Run Yosys analysis on Verilog code"""
    with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as tmp:
        tmp.write(verilog_code.encode('utf-8'))
        tmp_path = tmp.name
    
    try:
        # Run Yosys and capture output
        cmd = f"yosys -p 'read_verilog {tmp_path}; proc; opt; stat'"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=10)
        
        if process.returncode != 0:
            st.warning(f"Yosys warning: {stderr.decode('utf-8')}")
            return {}
            
        # Parse Yosys output
        output = stdout.decode('utf-8')
        stats = {}
        
        # Extract metrics
        for line in output.splitlines():
            if 'Number of wires' in line:
                stats['num_wires'] = int(line.split()[-1])
            elif 'Number of cells' in line:
                stats['num_cells'] = int(line.split()[-1])
            elif any(cell in line for cell in ['$dff', '$mux', '$not', '$and', '$or', '$xor']):
                parts = line.split()
                if len(parts) >= 2:
                    cell_type = parts[0][1:]  # Remove the $ prefix
                    stats[f"{cell_type}_count"] = int(parts[1])
                    
        return stats
    except subprocess.TimeoutExpired:
        st.warning("Yosys analysis timed out")
        return {}
    except Exception as e:
        st.warning(f"Error running Yosys: {str(e)}")
        return {}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def prepare_features_for_prediction(features, all_model_columns):
    """Prepare features for model prediction by ensuring all required columns are present"""
    # Create a DataFrame with a single row of features
    df = pd.DataFrame([features])
    
    # Ensure all columns required by the model are present
    for col in all_model_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only the columns that are in the model
    df = df[all_model_columns]
    
    return df

# User interface
tab1, tab2 = st.tabs(["Predict Circuit Depth", "About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio("Input Method", ["Paste Verilog Code", "Upload Verilog File"])
        
        if input_method == "Paste Verilog Code":
            verilog_code = st.text_area("Paste your Verilog code here:", height=300, 
                                        placeholder="module example(input clk, input reset, output reg [7:0] out);...")
        else:
            uploaded_file = st.file_uploader("Upload Verilog file", type=['.v'])
            if uploaded_file:
                verilog_code = uploaded_file.getvalue().decode('utf-8')
                st.code(verilog_code[:500] + "..." if len(verilog_code) > 500 else verilog_code, language="verilog")
            else:
                verilog_code = ""
    
    with col2:
        st.markdown("### Analysis Options")
        use_yosys = st.checkbox("Use Yosys (if available)", value=False, 
                               help="Enables advanced circuit analysis using Yosys. Requires Yosys to be installed.")
        
        selected_model = st.selectbox("Select Model", 
                                     ["GradientBoosting", "RandomForest", "XGBoost", "LinearRegression"],
                                     help="GradientBoosting generally has the best performance")
        
        compare_all = st.checkbox("Compare all models", value=False)
        
        st.markdown("### Example Verilog Codes")
        example_type = st.selectbox("Load Example", 
                                    ["", "Counter", "Combinational Logic", "Sequential Logic", "Memory"])
        
        if example_type == "Counter":
            verilog_code = """
module counter(input clk, input reset, output reg [3:0] count);
  always @(posedge clk or posedge reset) begin
    if (reset)
      count <= 4'b0000;
    else
      count <= count + 1;
  end
endmodule
"""
        elif example_type == "Combinational Logic":
            verilog_code = """
module adder(input [7:0] a, input [7:0] b, output [8:0] sum);
  assign sum = a + b;
endmodule
"""
        elif example_type == "Sequential Logic":
            verilog_code = """
module d_flip_flop(input clk, input d, output reg q);
  always @(posedge clk) begin
    q <= d;
  end
endmodule
"""
        elif example_type == "Memory":
            verilog_code = """
module ram(input clk, input we, input [7:0] addr, input [7:0] din, output reg [7:0] dout);
  reg [7:0] mem [0:255];
  
  always @(posedge clk) begin
    if (we)
      mem[addr] <= din;
    dout <= mem[addr];
  end
endmodule
"""
        
    if st.button("Predict Circuit Depth") and verilog_code and models:
        with st.spinner("Analyzing Verilog code..."):
            # Extract features
            basic_features = extract_basic_features(verilog_code)
            
            if use_yosys:
                try:
                    yosys_features = run_yosys_analysis(verilog_code)
                    features = {**basic_features, **yosys_features}
                    st.success("✅ Yosys analysis completed")
                except Exception as e:
                    st.error(f"⚠️ Error with Yosys analysis: {str(e)}")
                    features = basic_features
            else:
                features = basic_features
            
            top_module = get_top_module_name(verilog_code)
            
            # Display extracted features
            st.markdown("### Extracted Features")
            st.write(f"Top Module: {top_module}")
            
            # Determine module type
            module_type = next((k[len('module_type_'):] for k, v in features.items() 
                              if k.startswith('module_type_') and v == 1), "Unknown")
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Module Type", module_type)
            col2.metric("Signal Count", features.get('signal_count', 0))
            col3.metric("Logic Gates", sum([features.get(f'{op}_count', 0) 
                                          for op in ['and', 'or', 'not', 'xor']]))
            col4.metric("Control Blocks", features.get('if_count', 0) + features.get('case_count', 0))
            
            # Display other features
            with st.expander("View All Features"):
                st.json(features)
            
            # Get all model required columns from the first model
            first_model = next(iter(models.values()))
            model_columns = first_model.feature_names_in_ if hasattr(first_model, 'feature_names_in_') else None
            
            if model_columns is None:
                st.error("⚠️ Unable to determine required model features")
            else:
                # Prepare features for prediction
                X = prepare_features_for_prediction(features, model_columns)
                
                # Make predictions
                results = {}
                
                if compare_all:
                    st.markdown("### Model Predictions")
                    for name, model in models.items():
                        pred = model.predict(X)[0]
                        results[name] = pred
                    
                    # Display predictions
                    cols = st.columns(len(results))
                    for i, (name, pred) in enumerate(results.items()):
                        cols[i].metric(f"{name}", f"{pred:.2f}")
                    
                    # Plot comparison
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(results.keys(), results.values())
                    ax.set_ylabel("Predicted Circuit Depth")
                    ax.set_title("Comparison of Model Predictions")
                    st.pyplot(fig)
                    
                else:
                    # Just predict with selected model
                    pred = models[selected_model].predict(X)[0]
                    
                    # Display prediction with confidence
                    st.markdown("### Prediction Result")
                    st.metric("Predicted Circuit Depth", f"{pred:.2f}")
                    
                    # Add interpretation
                    depth_interpretation = ""
                    if pred < 5:
                        depth_interpretation = "This is a relatively simple circuit with low depth."
                    elif pred < 10:
                        depth_interpretation = "This circuit has moderate complexity."
                    else:
                        depth_interpretation = "This is a complex circuit with high depth."
                    
                    st.info(depth_interpretation)

with tab2:
    st.markdown("""
    ## About This Tool
    
    This web application predicts the depth of Verilog circuits before simulation using machine learning models.
    
    ### What is Circuit Depth?
    
    Circuit depth is a measure of the longest path through a digital circuit. It's an important metric for 
    understanding circuit performance, as it relates to the critical path and maximum clock frequency.
    
    ### How It Works
    
    1. **Feature Extraction**: The tool analyzes your Verilog code to extract features like:
       - Signal and instance counts
       - Logic operator usage
       - Control structures
       - Module classification
    
    2. **Prediction**: These features are fed into trained machine learning models to predict circuit depth.
    
    3. **Interpretation**: The results help you understand circuit complexity before simulation.
    
    ### Models
    
    Four regression models are available:
    - **Gradient Boosting**: Best overall performance (R² = 0.622)
    - **Random Forest**: Good generalization (R² = 0.611)
    - **XGBoost**: Solid performance (R² = 0.506)
    - **Linear Regression**: Baseline model (R² = -1.198)
    
    ### Limitations
    
    - The basic analysis doesn't use full synthesis, so predictions are approximations
    - Complex or non-standard Verilog code may not be analyzed correctly
    - The models were trained on a specific dataset and may not generalize to all circuit types
    
    ### Tips for Better Results
    
    - Use the Yosys option if available for more accurate feature extraction
    - Compare multiple model predictions to get a better understanding
    - Clean and well-structured Verilog code yields better analysis
    """)

# Add footer
st.markdown("---")
st.markdown("© 2025 Verilog Circuit Depth Predictor | Built with Streamlit")