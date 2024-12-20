# Fuel Cell Performance Prediction

This project uses machine learning models to predict `Target2` from a dataset of fuel cell performance metrics. The focus is on evaluating multiple models to determine the best one, optimizing both performance and interpretability.

## Features and Targets

- **Features:**
  - The dataset contains 15 features (`F1` to `F15`) used as predictors.
  
- **Target:**
  - `Target2`: The target variable for prediction.

## Models Used

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Support Vector Regressor (RBF Kernel)**
4. **Neural Network (MLP Regressor)**
5. **XGBoost Regressor**

## Metrics for Evaluation

- **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values. Lower is better.
- **R² Score:** Represents the proportion of variance in the target variable explained by the model. Higher is better (closer to 1).

## Results

Each model is evaluated based on the test dataset, and their performance is compared to identify the best model:

| Model                            | Mean Squared Error (MSE) | R² Score |
|----------------------------------|--------------------------|----------|
| Linear Regression               | 0.1062                   | 0.6356   |
| Random Forest Regressor         | 0.1116                   | 0.6169   |
| Support Vector Regressor (RBF)  | 0.1402                   | 0.5188   |
| Neural Network                  | 0.1715                   | 0.4115   |
| XGBoost Regressor               | 0.1206                   | 0.5861   |

**Best Model:** Linear Regression with an R² Score of 0.6356.

## Visualizations

1. **Model Performance Comparison:**
   - Bar plots showing MSE and R² scores for each model.
2. **Best Model Predictions vs. Actual Values:**
   - Scatterplot comparing predicted and actual values for the best-performing model.

## How to Run the Project

### 1. Prerequisites
- Python 3.8+
- Required libraries:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost`

Install dependencies using:
```bash
pip install pandas matplotlib seaborn scikit-learn xgboost
```

### 2. Dataset
- Place the dataset file `Fuel_cell_performance_data-Full.csv` in your working directory or update the file path in the code.

### 3. Running the Code
- Execute the script in a Jupyter Notebook, Python script, or Google Colab with GPU enabled for optimal performance.
- The code will:
  1. Load and preprocess the dataset.
  2. Split the data into training (70%) and testing (30%).
  3. Train and evaluate multiple models.
  4. Visualize the results.

### 4. Output
- Model evaluation metrics printed to the console.
- Visualizations of model performance and best model predictions.

## Project Structure

```
├── Fuel_cell_performance_data-Full.csv  # Dataset
├── fuel_cell_prediction.ipynb              # Main script
├── README.md                            # Project documentation
```

## Future Improvements

- Fine-tune hyperparameters for all models to achieve better accuracy.
- Implement additional models like Gradient Boosting or LightGBM.
- Explore feature engineering to enhance model performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the creators of the dataset and the open-source community for providing the tools to build this project.

---

**Contributions are welcome!** Feel free to open an issue or submit a pull request if you have suggestions or improvements.
