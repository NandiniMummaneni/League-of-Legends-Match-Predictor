**League of Legends Match Outcome Prediction Using PyTorch**
Part of IBM AI Engineering Professional Certificate — “Neural Networks and PyTorch”

This project demonstrates how to **build and evaluate a logistic regression model using PyTorch** to **predict the outcomes of League of Legends matches**.
It covers the complete machine learning pipeline — from data preprocessing and model training to optimization, visualization, and model interpretation.

**🚀 Project Overview :**

The project is divided into eight independent tasks, each focusing on a specific stage of model development:

**Task 1:** Data Loading and Preprocessing
Loads the League of Legends dataset.
Splits the data into features and target variables.
Standardizes the features and converts them into PyTorch tensors.

**Task 2:** Model Implementation
Defines a logistic regression model using torch.nn.Module.
Initializes the loss function and optimizer.

**Task 3:** Model Training and Evaluation
Implements a training loop with forward pass, loss computation, and backpropagation.
Evaluates model accuracy on both training and testing datasets.

**Task 4:** Model Optimization
Adds L2 regularization to improve generalization.
Retrains and compares the optimized model’s performance.

**Task 5:** Performance Visualization
Visualizes model performance using confusion matrix, ROC curve, and AUC score.
Generates a classification report to assess precision, recall, and F1-score.

**Task 6:** Model Saving and Loading
Saves the trained model’s state dictionary.
Reloads it into a new instance and verifies consistent performance.

**Task 7:** Hyperparameter Tuning
Tests multiple learning rates.
Identifies the best learning rate with the highest test accuracy.

**Task 8:** Feature Importance Analysis
Extracts weights from the trained model.
Sorts and visualizes feature importance to interpret which features influence predictions most.

**⚙️ Technologies Used :**
**PyTorch** – for model implementation and training and 
**Pandas, Scikit-learn** – for data processing and evaluation and 
**Matplotlib, Seaborn** – for visualization

**💡 This Project Demonstrates :**
1. A full machine learning workflow using PyTorch.
2. How optimization and regularization improve model performance.
3. How to visualize, interpret, and persist trained models.
4. A practical example of applying neural network fundamentals in a real dataset scenario.

