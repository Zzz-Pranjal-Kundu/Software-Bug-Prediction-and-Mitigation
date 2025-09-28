# Software Bug Prediction

A machine learning-based system for predicting software bugs using various classification algorithms and code metrics.

## 📋 Project Overview

This project implements a comprehensive bug prediction system that analyzes software code metrics to predict the likelihood of bugs in source code files. The system uses multiple machine learning classifiers and provides both individual and ensemble prediction capabilities.

## 🚀 Features

- **Multiple Classifiers**: Support for 9 different machine learning algorithms
- **Ensemble Learning**: Combines predictions from multiple models for improved accuracy
- **Comprehensive Metrics**: Evaluates models using precision, recall, F1-score, and accuracy
- **Cross-Validation**: Uses 10-fold cross-validation for robust model evaluation
- **Feature Analysis**: Analyzes importance of different code metrics in bug prediction

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
Install required packages:

```bash
pip install pandas scikit-learn numpy matplotlib seaborn

## 📁 Project Structure
software_bug_prediction/
├── classifiers/          # Individual classifier implementations
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── naive_bayes.py
│   ├── svm.py
│   ├── knn.py
│   ├── mlp.py
│   ├── ada_boost.py
│   ├── gradient_boosting.py
│   └── logistic_regression.py
├── ensemble.py          # Ensemble classifier combining all models
├── main.py             # Main execution script
├── metrics_analysis.py # Code metrics analysis utilities
└── README.md

## 🏗️ Classifiers Implemented
1. Decision Tree - DecisionTreeClassifier
2. Random Forest - RandomForestClassifier
3. Naive Bayes - GaussianNB
4. Support Vector Machine - SVC
5. K-Nearest Neighbors - KNeighborsClassifier
6. Multi-layer Perceptron - MLPClassifier
7. AdaBoost - AdaBoostClassifier
8. Gradient Boosting - GradientBoostingClassifier
9. Logistic Regression - LogisticRegression

## 📊 Usage

### Basic Execution

Run the main script to execute all classifiers:

```bash
python main.py

### Individual Classifiers

You can run individual classifiers directly:

```bash
python classifiers/decision_tree.py
python classifiers/random_forest.py
# ... etc for other classifiers

### Ensemble Model

Run the ensemble classifier:

```bash
python ensemble.py

### Metrics Analysis

Analyze code metrics and their importance:

```bash
python metrics_analysis.py

## 🔧 Configuration

The project uses scikit-learn's default parameters for most classifiers. Key configurations include:

 -Cross-validation: 10-fold
 -Test size: 20% (when train-test split is used)
 -Random state: 42 (for reproducibility)

## 📈 Model Evaluation

Each classifier is evaluated using:

 -Accuracy: Overall correctness
 -Precision: Quality of positive predictions
 -Recall: Coverage of actual positives
 -F1-Score: Harmonic mean of precision and recall

## 🎯 Input Data Format

The system expects a CSV file with the following structure:


Column	Description
Metric1	First code metric (e.g., cyclomatic complexity)
Metric2	Second code metric (e.g., lines of code)
...	...
MetricN	Nth code metric
bug	Target variable (0: no bug, 1: bug)

## 📝 Output

The system provides:

 -Individual classifier performance reports
 -Confusion matrices for each model
 -Classification metrics (precision, recall, F1-score, accuracy)
 -Ensemble model performance comparison
 -Feature importance analysis (for tree-based models)

## 🔬 Methodology

Data Preprocessing: Handles missing values and normalizes features

 -Model Training: Trains multiple classifiers using cross-validation
 -Evaluation: Comprehensive performance assessment
 -Ensemble: Combines predictions using voting or averaging
 -Analysis: Identifies most important code metrics for bug prediction

## 🤝 Contributing

Fork the repository

 -Create a feature branch (git checkout -b feature/amazing-feature)
 -Commit your changes (git commit -m 'Add some amazing feature')
 -Push to the branch (git push origin feature/amazing-feature)
 -Open a Pull Request