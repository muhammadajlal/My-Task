
# Anomaly Detection with One-Class SVM on Two Moons Dataset

This repository demonstrates anomaly detection using a One-Class SVM on a synthetic dataset generated from the "two moons" data. The project showcases how to create a training dataset containing only normal data and a test dataset that includes both normal and anomalous points, then train a One-Class SVM to detect anomalies.

## Overview

- **Dataset Generation:**  
  The dataset is created using scikit-learn's `make_moons` function. The data is split into:
  - **Training Data:** Consists solely of "normal" data points.
  - **Test Data:** Contains both normal and anomalous data points.
  
- **Model:**  
  The One-Class SVM is configured with an RBF kernel, using parameters `gamma=10` and `nu=0.001`. The model is trained exclusively on the normal training data.

- **Visualization:**  
  The project provides detailed visualizations:
  - A contour plot illustrating the decision function of the One-Class SVM.
  - Scatter plots that display the test data classified as "Ok data" (normal) and "Not ok data" (anomalies).
  - Consistent axis limits and styling to replicate an ideal anomaly detection view.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muhammadajlal/My-Task.git
   cd My-Task
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   **Dependencies include:**
   - numpy
   - matplotlib
   - scikit-learn

## Usage

Run the main script to generate the dataset, train the model, and display the anomaly detection results:

```bash
python IAV_Task.py
```

## Results

The output visualization includes:
- Scatter plots show the overall structure of training and test data and the desired outcome of our anomaly detection model. 
- Scatter plot showing the result of our anomaly detection dataset, with blue markers for "Ok data" (normal) and red markers for "Not ok data" (anomalies).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
