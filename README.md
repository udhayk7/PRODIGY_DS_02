# Titanic Survival Analysis 

## Project Overview
This data science project performs an in-depth exploratory data analysis (EDA) of the famous Titanic dataset. The analysis explores various factors that influenced passenger survival rates during the Titanic disaster.

## Key Features
- Comprehensive data cleaning and preprocessing
- Detailed statistical analysis
- Interactive visualizations using Seaborn and Matplotlib
- Survival rate analysis based on:
  * Gender
  * Passenger Class
  * Age Groups
  * Family Size
- Correlation analysis of numerical variables

## Dataset Information
The analysis uses the Titanic dataset from Kaggle, which includes:
- 891 passengers
- 12 variables including age, sex, passenger class, fare, etc.
- Binary survival outcome (0 = No, 1 = Yes)

## Project Structure
```
PRODIGY_DS_02/
├── data/               # Dataset directory
│   └── train.csv      # Titanic training dataset
├── titanic_analysis.py # Main analysis script
├── titanic_analysis.ipynb # Jupyter notebook version
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Key Insights
1. **Gender Impact**: Strong correlation between gender and survival
2. **Class Influence**: Higher passenger classes had better survival rates
3. **Age Patterns**: Children had higher survival rates
4. **Family Size**: Passengers traveling with family showed different survival patterns

## Technologies Used
- Python 3.12
- Libraries:
  * pandas (≥2.1.0)
  * numpy (≥1.26.0)
  * matplotlib (≥3.8.0)
  * seaborn (≥0.13.0)
  * scikit-learn (≥1.3.0)
  * Jupyter Lab

## Setup and Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd PRODIGY_DS_02
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
- Using Python script:
```bash
python titanic_analysis.py
```
- Using Jupyter Notebook:
```bash
jupyter lab
```

## Visualizations
The project includes several visualizations:
- Bar plots for survival rates by different categories
- Box plots for age distribution analysis
- Correlation heatmap for numerical variables
- Distribution plots for continuous variables

## Future Enhancements
- Machine learning model development
- Feature engineering
- Interactive dashboard creation
- Additional demographic analysis

## Contributing
Feel free to fork this repository and submit pull requests. You can also open issues for any bugs or suggestions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Part of the Prodigy InfoTech Data Science Internship Program
