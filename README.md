🎯 Customer Analytics, Churn Prediction & Movie Recommendation System

This project combines customer analytics, churn prediction, and a content-based movie recommendation system into a unified machine-learning pipeline. It processes customer activity data, identifies behavioral segments, predicts which users are likely to churn, and finally recommends Top-N movies based on each user's interests.

🚀 Project Objectives
🔹 1. Customer Analytics

Understand user behavior using aggregated transactional and engagement features.

🔹 2. Customer Segmentation

Cluster users into meaningful behavioral groups using unsupervised learning.

🔹 3. Churn Labeling & Prediction

Define churn using an anchored time-window and train ML models to predict churn probability.

🔹 4. Personalized Movie Recommendation

Recommend Top-N movies for each user based on their historical preferences and similarity scoring.
🧹 Data Processing & Feature Engineering
✔ Preprocessing Steps

1)Handling missing values

2)Cleaning and normalizing numeric features

3)Parsing timestamps & generating user activity metrics

4)Merging multiple datasets into a unified user-profile table

✔ Engineered Features

1)Monthly spend

2)Average order value

3)Total transactions

4)Days active

5)Session gaps and recency scores

6)Rolling-window activity aggregates

These features enable better segmentation and churn modeling.

🧭 Customer Segmentation
🛠 Techniques:

1)K-Means Clustering

2)Elbow Method for optimal k

3)Silhouette Score for validation

Cluster profiling

🎯 Outcomes:

Clusters represent user groups such as:

1)High-value power users

2)Low-engagement at-risk users

3)Moderate spenders with stable activity

🔔 Churn Definition & Labeling

Churn is defined using:

1)Anchored activity windows

2)Time gaps between user sessions

3)"No activity beyond threshold days" ⇒ Churn = 1

4)This produces a labeled dataset for supervised learning.

🤖 Churn Prediction Models
Models Used:

1)Logistic Regression

2)Random Forest

3)Gradient Boosting (XGBoost / LightGBM optional)

4)Class Imbalance Handling:

5)Random oversampling

6)SMOTE (if applicable)

Evaluation Metrics:

1)Accuracy

2)Recall (important for churn)

3)F1-Score

4)Confusion Matrix

5)Feature importance plots

🎬 Movie Recommendation System

The notebook also includes a Content-Based Movie Recommendation System.

🔍 How It Works

Movie metadata is vectorized using TF-IDF or embedding techniques

User preferences are calculated based on liked/watched movies

Cosine similarity identifies movies closest to user interests

🎯 Output

For each user, the system generates:

Top-N recommended movies tailored to their interests

This can be easily integrated into a real product dashboard.

🛠 Tech Stack

1)Python 3

2)Pandas, NumPy

3)scikit-learn

4)NLTK / TF-IDF Vectorizer

5)Matplotlib / Seaborn

Jupyter Notebook

▶ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Open the notebook
jupyter notebook "ba-project-code.ipynb"

3. Run sections in order

1)Data preprocessing

2)Feature engineering

Clustering

1)Churn labeling & prediction

2)Movie recommendation system

📈 Key Insights & Deliverables

1)Behavioral clusters revealing different user groups

2)Predictive churn model with interpretability

3)Personalized movie recommendation engine capable of producing Top-N suggestions

4)Modular notebook enabling easy adaptation for business use cases

🔮 Future Enhancements

1)Hybrid collaborative + content-based recommender

2)Real-time API deployment for churn scores

3)User dashboard for personalized insights.

✍ Author
Tapan Datta
