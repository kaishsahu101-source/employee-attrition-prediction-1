# Employee Attrition Prediction System

A complete machine learning project that predicts whether an employee is likely to leave a company (attrition) using historical employee data.

## 🎯 Objectives

- Predict employee attrition (Yes / No)
- Identify key factors that influence attrition
- Build a simple dashboard to visualize insights
- Deploy the model as a web app

## 👥 Target Users

- **HR teams** - For decision making and retention strategies
- **Managers** - To monitor team health and engagement
- **Students** - To learn ML workflow end-to-end

## 📊 Dataset

The project uses the "10000 HRA Records.csv" dataset containing employee information with the following features:

### Employee Information
- Age, Gender, Education, Marital Status
- Department, Job Role, Job Level
- Years at Company, Years in Current Role
- Years Since Last Promotion, Years with Current Manager

### Compensation
- Monthly Income, Percent Salary Hike, Stock Option Level

### Work Conditions
- Work-Life Balance, Job Satisfaction
- Environment Satisfaction, Relationship Satisfaction
- OverTime (Yes/No), Distance From Home

### Performance
- Performance Rating, Training Times Last Year

### Target Variable
- **Attrition** - Yes / No (whether employee left the company)

## 🚀 Quick Start

### 1. Clone the Dataset Repository
```bash
git clone https://github.com/kaishsahu101-source/employee-attrition-prediction-1.git
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib flask
```

### 3. Train the Model
```bash
cd employee-attrition-project
python train_model.py
```

### 4. Run the Web Application
```bash
python app.py
```

Then open your browser and navigate to: `http://localhost:5001`

## 📁 Project Structure

```
employee-attrition-project/
├── train_model.py          # Data preprocessing and model training
├── app.py                  # Flask web application
├── models/
│   └── attrition_model.pkl # Trained model (generated after training)
├── templates/
│   └── index.html          # Web interface
├── static/                 # Static files (CSS, JS, images)
└── README.md               # This file
```

## 🔧 Features

### ✅ Data Preprocessing
- Handle missing values (fill with median/mode)
- Convert categorical data into numbers (Label Encoding)
- Normalize/scale numerical values (StandardScaler)
- Split dataset into training (80%) and testing (20%) sets

### ✅ Model Training
- **Logistic Regression** - Baseline model
- **Random Forest** - Ensemble model (typically performs better)
- Automatic selection of best performing model

### ✅ Model Evaluation
- Accuracy
- Precision
- Recall
- Confusion Matrix
- Feature Importance Analysis

### ✅ Prediction System
- User inputs employee details through a web form
- Model predicts: Attrition Risk (Yes / No)
- Probability Score (e.g., 75% = high risk)
- Risk Level Classification (High/Medium/Low)

### ✅ Feature Importance
Display which factors most affect attrition:
- Overtime → High impact
- Job Satisfaction → Critical factor
- Years at Company → Experience matters
- Work-Life Balance → Employee wellbeing
- Monthly Income → Compensation fairness

## 📈 Expected Results

The model typically achieves:
- **Accuracy**: 70-85%
- **Precision**: Good balance for HR use cases
- **Recall**: Identifies most at-risk employees

## 🌐 Web Interface

The Flask application provides:
- Beautiful, responsive UI
- Easy-to-use prediction form
- Real-time predictions
- Risk level visualization
- Educational content about the system

## 🔬 API Endpoint

For programmatic access, use the REST API:

```bash
POST http://localhost:5001/api/predict
Content-Type: application/json

{
    "Age": 30,
    "Gender": "Male",
    "Department": "Sales",
    "JobRole": "Sales Executive",
    "MonthlyIncome": 5000,
    "YearsAtCompany": 3,
    "JobSatisfaction": 3,
    "WorkLifeBalance": 3,
    "OverTime": "No",
    "DistanceFromHome": 5,
    "PercentSalaryHike": 15,
    "YearsInCurrentRole": 2
}
```

Response:
```json
{
    "prediction": "No",
    "probability": 25.5,
    "risk_level": "Low"
}
```

## 💡 Usage Tips

1. **For HR Teams**: Use the probability scores to prioritize intervention efforts
2. **For Managers**: Monitor team members with high risk scores
3. **For Students**: Study the code to understand the complete ML pipeline

## 🎓 Learning Outcomes

By studying this project, you'll learn:
- Data preprocessing techniques
- Classification algorithms
- Model evaluation metrics
- Feature importance analysis
- Building ML-powered web applications
- End-to-end ML project workflow

## 📝 License

This project is for educational purposes. Feel free to use and modify.

## 🤝 Contributing

Suggestions and improvements are welcome!

---

**Built with ❤️ using Python, Scikit-learn, and Flask**
