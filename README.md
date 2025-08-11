# 1985 Automobile Dataset - Price Prediction Project

*By Shubh Gupta*

## 📋 Project Overview

This project explores the classic 1985 Automobile Dataset to build machine learning models that can accurately predict automobile prices based on technical specifications and features. The project demonstrates comprehensive data analysis, preprocessing techniques, and model comparison using various machine learning algorithms.

## 🎯 Project Goals

- Analyze the relationship between automobile features and pricing
- Implement and compare multiple machine learning models for price prediction
- Demonstrate best practices in data preprocessing and feature engineering
- Provide insights into the 1985 automotive market through exploratory data analysis

## 📊 Dataset Description

The dataset contains **201 automobile records** with **29 features** covering various aspects of vehicle design and performance from the 1985 automotive market.

### Key Statistics
- **Price Range**: $5,118 - $45,400
- **Average Price**: $13,207
- **Total Features**: 29 (18 numerical, 10 categorical)
- **Top Manufacturers**: Toyota (32 models), Nissan (18), Mazda (17), Mitsubishi (13), Honda (13)

### Feature Categories

**Numerical Features (18):**
- Physical dimensions: wheel-base, length, width, height, curb-weight
- Engine specifications: engine-size, bore, stroke, compression-ratio, horsepower, peak-rpm
- Performance metrics: city-mpg, highway-mpg, city-L/100km
- Safety: normalized-losses, symboling
- Fuel type indicators: diesel, gas

**Categorical Features (10):**
- Make and model information
- Body style (sedan, hatchback, wagon, convertible, hardtop)
- Engine configuration (aspiration, engine-type, num-of-cylinders, fuel-system)
- Drive train (drive-wheels, engine-location, num-of-doors)

## 🏗️ Methodology

### Data Preprocessing Pipeline

1. **Data Cleaning**
   - Handled missing values using appropriate imputation strategies
   - Normalized numerical features to ensure fair comparison
   - Encoded categorical variables for machine learning compatibility

2. **Feature Engineering**
   - Created derived features like `city-L/100km` for fuel efficiency
   - Binned horsepower into categories (Low, Medium, High)
   - Added fuel type indicators (diesel vs gas)

3. **Preprocessing Pipeline**
   - **Numerical features**: Mean imputation + standardization
   - **Categorical features**: Mode imputation + one-hot encoding

### Machine Learning Models

The project implements and compares three different machine learning algorithms:

1. **Gradient Boosting Regressor**
   - RMSE: $2,342.99
   - MAE: $1,490.09
   - Performance: Best overall performance
   - Advantages: Handles non-linear relationships well, robust to outliers

2. **Random Forest Regressor**
   - RMSE: $3,193.12
   - MAE: $1,950.25
   - Performance: Second best performance
   - Advantages: Provides feature importance, handles mixed data types

3. **Support Vector Machine (SVM)**
   - RMSE: $11,104.09
   - MAE: $7,041.45
   - Performance: Least effective for this dataset
   - Limitations: Linear kernel may be too restrictive for complex price relationships

## 📈 Results & Insights

### Model Performance Comparison

| Model | RMSE | MAE | Performance Rank |
|-------|------|-----|------------------|
| Gradient Boosting | $2,343 | $1,490 | 1st |
| Random Forest | $3,193 | $1,950 | 2nd |
| SVM (Linear) | $11,104 | $7,041 | 3rd |

### Key Findings

- **Horsepower vs Price**: Strong positive correlation between engine power and vehicle price
- **Manufacturer Impact**: Toyota dominates the dataset with 32 different models
- **Body Style Influence**: Different body styles show varying price distributions
- **Feature Importance**: Engine specifications and physical dimensions are key predictors

## 📁 Project Structure

```
1985 Automobile Dataset/
├── auto.csv                    # Original dataset
├── auto_clean.csv             # Preprocessed dataset
├── autoModel.ipynb            # Main analysis notebook
├── automobile_price_prediction_blog.md  # Detailed project report
├── feature_importance.png     # Feature importance visualization
├── model_performance.png      # Model comparison chart
├── price_by_body_style.png   # Price distribution by body style
├── price_distribution.png     # Overall price distribution
├── price_vs_horsepower.png   # Price vs horsepower relationship
├── top_manufacturers.png      # Manufacturer distribution
├── Project Report Shubh Gupta.docx  # Project report (Word format)
├── Project Report Shubh Gupta.pdf   # Project report (PDF format)
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see requirements below)

### Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open `autoModel.ipynb` in Jupyter Notebook

### Usage

1. **Data Exploration**: Run the notebook cells to explore the dataset
2. **Model Training**: Execute the machine learning pipeline to train models
3. **Results Analysis**: View generated visualizations and performance metrics
4. **Custom Analysis**: Modify the notebook for your own analysis

## 📊 Generated Visualizations

The project generates several key visualizations:

- **Price Distribution**: Shows the overall distribution of automobile prices
- **Top Manufacturers**: Displays the most common car manufacturers
- **Price vs Horsepower**: Illustrates the relationship between engine power and price
- **Price by Body Style**: Shows price variations across different vehicle body styles
- **Model Performance**: Compares the performance of different ML models
- **Feature Importance**: Highlights the most important features for price prediction

## 🔧 Technical Details

### Data Preprocessing
- **Missing Value Handling**: Mean imputation for numerical, mode for categorical
- **Feature Scaling**: StandardScaler for numerical features
- **Categorical Encoding**: OneHotEncoder for categorical variables

### Model Training
- **Train-Test Split**: 80-20 split with random state 42
- **Cross-Validation**: Not implemented in current version
- **Hyperparameter Tuning**: Basic models without optimization

### Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy in dollars
- **MAE (Mean Absolute Error)**: Provides absolute error interpretation

## 📚 Future Improvements

- Implement cross-validation for more robust model evaluation
- Add hyperparameter tuning for better model performance
- Explore deep learning approaches (neural networks)
- Create a web application for interactive price prediction
- Add more recent automotive datasets for comparison

## 👨‍💻 Author

**Shubh Gupta**
- Student at Boston University
- Course: CS677 - Data Science with Python
- Semester: Fall 2023

## 📄 License

This project is created for educational purposes as part of a university course assignment.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome. Please feel free to:
- Report issues or bugs
- Suggest new features or analyses
- Improve the documentation
- Enhance the code quality

## 📞 Contact

For questions or feedback about this project, please reach out through the course instructor or create an issue in the repository.

---

*Last updated: December 2023* 