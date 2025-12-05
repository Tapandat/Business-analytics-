# Customer Intelligence Platform

A comprehensive analytics platform for customer behavior analysis, churn prediction, and personalized recommendations using advanced machine learning techniques.

## 🎯 Overview

This platform helps businesses understand customer behavior, predict churn risk, and deliver personalized experiences. It combines multiple ML models including churn prediction, customer segmentation, and collaborative filtering recommendations with explainable AI (SHAP) for transparency.

## ✨ Features

### 1. User Lookup Dashboard
- **Churn Risk Assessment**: Predict which customers are likely to leave with probability scores
- **Customer Lifetime Value (CLV)**: Calculate and visualize long-term customer value
- **SHAP Explanations**: Understand exactly why the model predicts a customer will churn
- **Personalized Recommendations**: Generate AI-powered movie suggestions for each user

### 2. Cluster Insights
- **Customer Segmentation**: Unsupervised clustering to identify distinct customer groups
- **Cluster Personas**: Detailed profiles with behavioral and financial metrics
- **Value at Risk Heatmap**: Identify high-value customers at risk of churning
- **Radar Plots**: Visual comparison of segment characteristics
- **Segment Analysis**: Compare size, churn rates, and value across segments

### 3. Model Performance
- **Model Comparison**: Evaluate multiple ML algorithms side-by-side
- **ROC & PR Curves**: Visualize model discrimination ability
- **Threshold Analysis**: Interactive confusion matrix at different cutoffs
- **Feature Importance**: Identify key drivers of churn predictions
- **Recommendation Stats**: Analyze recommendation system coverage and quality

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Models**: 
  - scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
  - LightGBM
  - XGBoost
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Recommendations**: Surprise (collaborative filtering)
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## 📋 Prerequisites

- Python 3.11
- pip package manager

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd final_ba_fin
```

2. **Install required packages**
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn lightgbm scikit-surprise
```

Note: For SHAP functionality, you may need to manage numpy versions:
```bash
pip install "numpy<2"  # Required for scikit-surprise compatibility
```

## 📊 Data Files Required

The application expects the following files in the project directory:

### Core Data Files
- `user_features.csv` - User profile and behavioral features
- `churn_scored.csv` - Churn predictions and CLV calculations
- `cluster_personas.csv` - Customer segment profiles
- `churn_features_processed.csv` - Processed features for SHAP analysis

### Model Files
- `best_churn_model.pkl` - Trained churn prediction model
- `churn_feature_columns.pkl` - Feature column names
- `shap_explainer.pkl` - SHAP explainer object
- `recommender_model.pkl` - Recommendation model configuration
- `ratings_data.csv` - User-movie ratings for recommendations

### Optional Files
- `model_comparison.csv` - Model performance comparison metrics
- `test_predictions.csv` - Test set predictions for ROC/PR curves

## 🎮 Usage

1. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

2. **Navigate through the platform**
   - Start with the **About** page to understand the platform
   - Use **User Lookup Dashboard** to analyze individual customers
   - Explore **Cluster Insights** to understand customer segments
   - Review **Model Performance** to evaluate prediction quality

## 📓 Jupyter Notebook

The `ba-project-take2.ipynb` notebook contains:
- Data preprocessing and feature engineering
- Model training and evaluation
- SHAP analysis
- Customer segmentation (K-Means clustering)
- Recommendation system training
- Data export for Streamlit app

Run the notebook to generate all required data files before using the Streamlit app.

## 🔑 Key Metrics Explained

### Churn Prediction
- **Churn Probability**: Likelihood (0-1) that a customer will leave
- **Churn Risk Band**: Low/Medium/High risk categorization
- **Expected Value Loss**: CLV × Churn Probability (potential revenue at risk)

### Customer Value
- **CLV (Customer Lifetime Value)**: Total expected revenue from a customer
- **CLV Band**: Low/Medium/High value categorization
- **Priority Segment**: Combination of CLV and churn risk for targeting

### Model Performance
- **ROC-AUC**: Area under ROC curve (0.5-1.0, higher is better)
- **PR-AUC**: Area under Precision-Recall curve
- **Accuracy**: Percentage of correct predictions
- **Recall**: Percentage of actual churners correctly identified

## 🎨 Customization

### Modifying Thresholds
Edit the classification threshold in the Model Performance dashboard to adjust the precision-recall trade-off.

### Adding New Features
1. Add features in the notebook's feature engineering section
2. Retrain models
3. Export updated data files
4. Features will automatically appear in the Streamlit app

### Changing Cluster Count
Modify the `best_k` parameter in the notebook's clustering section to change the number of customer segments.

## 🐛 Troubleshooting

### Model Loading Errors
If you see "model compatibility issues":
- Re-run the notebook to regenerate model files with your current environment
- Ensure numpy version compatibility (`numpy<2` for scikit-surprise)

### Missing Data Files
If data files are not found:
- Run all cells in the notebook to generate required files
- Check that files are in the same directory as `streamlit_app.py`

### SHAP Not Working
- Ensure `shap` package is installed: `pip install shap`
- Verify `shap_explainer.pkl` and `churn_features_processed.csv` exist
- Check numpy version compatibility

### Recommendations Not Loading
- Ensure `scikit-surprise` is installed
- Verify `ratings_data.csv` and `recommender_model.pkl` exist
- Check that numpy version is `<2`

## 📈 Business Impact

This platform enables:
- **Proactive Churn Prevention**: Identify at-risk customers before they leave
- **Targeted Retention**: Focus efforts on high-value customers
- **Personalization at Scale**: Deliver tailored experiences to thousands of users
- **Data-Driven Decisions**: Base strategies on ML predictions, not intuition
- **ROI Optimization**: Maximize return on retention and marketing spend

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ML models (Neural Networks, Ensemble methods)
- Real-time prediction API
- A/B testing framework
- Advanced recommendation algorithms (Deep Learning, Hybrid models)
- Automated model retraining pipeline

## 📄 License

This project is for educational and demonstration purposes.

## 👥 Authors
Tapan Datta 
Business Analytics Project - Customer Intelligence Platform

## 🙏 Acknowledgments

- Streamlit for the amazing web framework
- SHAP library for explainable AI
- Surprise library for collaborative filtering
- scikit-learn community for ML tools

---

**Note**: This is a demonstration project. For production use, consider:
- Database integration for real-time data
- Model versioning and monitoring
- API endpoints for predictions
- Authentication and authorization
- Scalability optimizations
- Automated testing and CI/CD
