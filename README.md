# Used-Cars-Price-Prediction-Model
This project is a part of my data analysis internship at CodeAlpha.
# Car Price Prediction Project

This repository contains a Python script for predicting car prices using machine learning models. The project involves data analysis, visualization, and the implementation of both Linear Regression and Random Forest Regression models to predict the selling price of cars based on various features.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict the selling price of cars using a dataset containing various features such as car age, present price, kilometers driven, and car model. The project includes:

- **Data Loading and Preprocessing**: Loading the dataset, checking for null values, and calculating car age.
- **Data Visualization**: Creating various plots to understand the distribution of selling prices and relationships between different features.
- **Model Training and Evaluation**: Implementing and evaluating Linear Regression and Random Forest Regression models.
- **Model Saving**: Saving the trained model for future use.

## Features

- **Data Analysis**: The script includes exploratory data analysis (EDA) to understand the dataset's structure and relationships between variables.
- **Data Visualization**: The script generates several plots, including histograms, scatter plots, and bar plots, to visualize the data.
- **Machine Learning Models**: The script implements and evaluates two machine learning models:
  - **Linear Regression**
  - **Random Forest Regression**
- **Model Evaluation**: The script calculates and prints evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) score.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Run the script**:

   ```bash
   python Price_Prediction.py
   ```

3. **View the results**:

   The script will output the evaluation metrics for both models and save the trained model to a file named `car data.csv`.

## Results

The script outputs the following evaluation metrics for each model:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²) Score**

These metrics help in understanding the performance of the models in predicting car prices.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/your-profile/) for any questions or discussions related to this project! 🚗💻
