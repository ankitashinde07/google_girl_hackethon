Overview
This project automates the analysis of combinational logic depth in RTL (Register Transfer Level) designs using Machine Learning (ML). Instead of running full synthesis, which is slow, this tool predicts logic depth based on extracted circuit features.

Why This Project?
✅ Speeds up timing analysis by predicting logic depth before synthesis.
✅ Helps FPGA & ASIC designers optimize circuits early.
✅ Uses ML models (Random Forest, XGBoost, Gradient Boosting) for fast predictions.

Project Workflow
1️⃣ Feature Extraction from Verilog Code (parser.py)
2️⃣ Convert Extracted Data to CSV (csv_convertor.py)
3️⃣ Train ML Models (ml_model.ipynb)
4️⃣ Predict Combinational Depth (predict_depth.py)

Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/ankitashinde07/google_girl_hackethon.git
cd google_girl_hackethon
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Feature Extraction
python src/parser.py --input rtl_design.v
4️⃣ Convert to CSV
python src/csv_convertor.py
5️⃣ Train the ML Model
jupyter notebook ml_model.ipynb
6️⃣ Predict Combinational Depth
python src/predict_depth.py --input rtl_design.v

Project Structure
📁 dataset/ → Contains training data (modules.csv).
📁 src/ → Code for feature extraction, ML training, and prediction.
📁 models/ → Saved ML models (XGBoost.pkl, RandomForest.pkl, etc.).
📁 examples/ → Sample Verilog files for testing.
📄 README.md → Project documentation.

Machine Learning Models Used
🟢 Random Forest
🟢 Gradient Boosting
🟢 XGBoost
🟢 Linear Regression

The model is trained on extracted RTL features (fan-in, fan-out, gate count) to predict combinational depth.
Example Prediction Output
{
  "Circuit": "alu.v",
  "Fan-In": 3,
  "Fan-Out": 2,
  "Gate Count": 120,
  "Predicted Combinational Depth": 5
}

Future Improvements
✅ Expand dataset with real FPGA/ASIC designs.
✅ Compare Neural Networks for prediction accuracy.
✅ Deploy as a web app for easy use.

License
📜 This project is open-source under the MIT License.

