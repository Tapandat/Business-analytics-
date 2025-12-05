import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# HELPER FUNCTIONS (CACHED)
# =========================

@st.cache_data
def load_user_data():
    """Load user features, churn scores, and segmentation data."""
    try:
        user_features = pd.read_csv("user_features.csv")
        churn_scored = pd.read_csv("churn_scored.csv")
        cluster_personas = pd.read_csv("cluster_personas.csv")
        return user_features, churn_scored, cluster_personas
    except FileNotFoundError:
        st.warning("Data files not found. Please run your notebook to generate them.")
        return None, None, None

@st.cache_data
def load_processed_features():
    """Load the processed feature matrix used for churn model training."""
    try:
        churn_features_processed = pd.read_csv("churn_features_processed.csv")
        return churn_features_processed
    except FileNotFoundError:
        return None

@st.cache_resource
def load_churn_model():
    """Load the trained churn model."""
    try:
        with open("best_churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Churn model not found. Please train and save it from your notebook.")
        return None

@st.cache_resource
def load_shap_explainer():
    """Load SHAP explainer."""
    try:
        with open("shap_explainer.pkl", "rb") as f:
            explainer = pickle.load(f)
        return explainer
    except (FileNotFoundError, ModuleNotFoundError):
        return None

@st.cache_resource
def load_recommender():
    """Load the recommendation model configuration and retrain it."""
    try:
        from surprise import Dataset, Reader, KNNWithMeans, KNNBasic, SVD, BaselineOnly
        import pickle
        
        # Load the model configuration
        with open("recommender_model.pkl", "rb") as f:
            rec_artifacts = pickle.load(f)
        
        # Load ratings data
        ratings_df = pd.read_csv("ratings_data.csv")
        
        # Rebuild the dataset and trainset
        reader = Reader(rating_scale=(ratings_df["rating"].min(), ratings_df["rating"].max()))
        data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        
        # Reconstruct the model based on saved configuration
        model_name = rec_artifacts.get("model_name", "ItemKNNWithMeans")
        model_params = rec_artifacts.get("model_params", {})
        
        # Create model instance based on name with memory-efficient settings
        if "KNNWithMeans" in model_name:
            # Use memory-efficient settings: limit k neighbors and use min_support
            sim_options = {
                "name": "cosine",
                "user_based": False,
                "min_support": 3  # Require at least 3 common ratings
            }
            model = KNNWithMeans(k=40, sim_options=sim_options)  # Limit to 40 neighbors
        elif "KNN" in model_name and "user" in model_name.lower():
            sim_options = {"name": "cosine", "user_based": True, "min_support": 3}
            model = KNNBasic(k=40, sim_options=sim_options)
        elif "KNN" in model_name:
            sim_options = {"name": "cosine", "user_based": False, "min_support": 3}
            model = KNNBasic(k=40, sim_options=sim_options)
        elif "SVD" in model_name:
            model = SVD(n_factors=50, n_epochs=20, random_state=42)
        else:
            # Default to SVD (more memory efficient than KNN)
            model = SVD(n_factors=50, n_epochs=20, random_state=42)
        
        # Train the model
        model.fit(trainset)
        
        return {
            "model": model,
            "trainset": trainset,
            "all_movies": rec_artifacts.get("all_movies", []),
            "all_users": rec_artifacts.get("all_users", []),
            "ratings_df": ratings_df
        }
    except FileNotFoundError:
        return None
    except ModuleNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading recommender: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

@st.cache_data
def load_movies_data():
    """Load movies metadata if available."""
    try:
        movies = pd.read_csv("movies.csv")
        return movies
    except FileNotFoundError:
        return None

@st.cache_data
def load_feature_columns():
    """Load the feature columns used for churn model."""
    try:
        with open("churn_feature_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        return feature_cols
    except FileNotFoundError:
        return None

# =========================
# MODEL PERFORMANCE DASHBOARD
# =========================

def show_model_performance():
    st.markdown('<div class="main-header">📈 Model Performance Dashboard</div>', unsafe_allow_html=True)
    
    st.write("""
    **Purpose:** Evaluate and compare machine learning models to ensure reliable predictions.
    This dashboard shows model accuracy, discrimination ability, and recommendation system quality.
    """)
    
    # Load churn model and data (handle errors gracefully)
    try:
        churn_model = load_churn_model()
    except Exception as e:
        churn_model = None
        # Don't show warning here - will show in Feature Importance tab only
    
    processed_features = load_processed_features()
    
    if processed_features is None:
        st.error("❌ Processed features data not loaded.")
        st.info("Please ensure the processed features are saved from your notebook.")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison",
        "📈 Performance Curves",
        "🎯 Feature Importance",
        "🎬 Recommendation Stats"
    ])
    
    # Tab 1: Model Comparison
    with tab1:
        st.markdown("### 📊 Churn Model Performance Comparison")
        st.write("🏆 **Overview:** Compare multiple machine learning algorithms to identify the best performer.")
        st.caption("Models are evaluated on ROC-AUC (discrimination ability) and PR-AUC (precision-recall balance). Higher scores = better performance.")
        
        # Try to load saved comparison data
        try:
            model_comparison = pd.read_csv("model_comparison.csv")
            
            # Display table
            st.dataframe(model_comparison.sort_values("roc_auc", ascending=False), use_container_width=True, hide_index=True)
            
            # Visualize metrics
            st.markdown("### 📊 Metric Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC-AUC comparison
                fig, ax = plt.subplots(figsize=(8, 5))
                order = model_comparison.sort_values("roc_auc", ascending=False)["model"]
                sns.barplot(data=model_comparison, x="model", y="roc_auc", order=order, ax=ax, palette="viridis")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_ylabel("ROC-AUC Score")
                ax.set_xlabel("Model")
                ax.set_title("ROC-AUC by Model")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # PR-AUC comparison
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=model_comparison, x="model", y="pr_auc", order=order, ax=ax, palette="plasma")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_ylabel("PR-AUC Score")
                ax.set_xlabel("Model")
                ax.set_title("PR-AUC by Model")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Best model highlight
            best_model_row = model_comparison.sort_values("roc_auc", ascending=False).iloc[0]
            st.success(f"🏆 **Best Model:** {best_model_row['model']} with ROC-AUC: {best_model_row['roc_auc']:.4f}")
            
        except FileNotFoundError:
            st.warning("Model comparison data not found. Please save it from your notebook.")
            st.info("""
            **Note:** To display actual model comparison metrics, save the model comparison results from your notebook:
            
            ```python
            # In your notebook, after evaluating all models:
            summary_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
            ```
            """)
    
    # Tab 2: Performance Curves
    with tab2:
        st.markdown("### 📈 ROC and Precision-Recall Curves")
        st.write("📉 **Overview:** Visualize model performance across all classification thresholds.")
        st.caption("ROC curve shows true positive vs false positive rate. PR curve shows precision vs recall trade-off. Use threshold slider to see confusion matrix at different cutoffs.")
        
        try:
            test_preds = pd.read_csv("test_predictions.csv")
            
            from sklearn.metrics import roc_curve, precision_recall_curve, auc
            
            y_true = test_preds["y_true"]
            y_proba = test_preds["y_proba"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
                ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Receiver Operating Characteristic (ROC) Curve")
                ax.legend(loc="lower right")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = auc(recall, precision)
                
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.plot(recall, precision, color="green", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve")
                ax.legend(loc="lower left")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Confusion matrix at different thresholds
            st.markdown("### 🎯 Threshold Analysis")
            threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
            
            from sklearn.metrics import confusion_matrix, classification_report
            
            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix\n(threshold={threshold:.2f})")
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)
        
        except FileNotFoundError:
            st.warning("Test predictions not found. Please save them from your notebook.")
            st.info("""
            **Note:** To display ROC and PR curves, you need to save predictions from your notebook:
            
            ```python
            # Save test predictions for visualization
            test_predictions = pd.DataFrame({
                'y_true': y_test,
                'y_proba': best_model.predict_proba(X_test)[:, 1]
            })
            test_predictions.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
            ```
            """)
    
    # Tab 3: Feature Importance (SHAP-based Global)
    with tab3:
        st.markdown("### 🎯 Global Feature Importance (SHAP)")
        st.write("🔍 **Overview:** Understand which customer features have the strongest impact on churn predictions.")
        st.caption("Feature importance helps identify what drives customer churn, enabling targeted interventions.")
        
        st.info("💡 For user-specific feature importance, visit **User Lookup Dashboard** → Select a user → **SHAP Explanation** tab")
        
        
    # Tab 4: Recommendation Stats
    with tab4:
        st.markdown("### 🎬 Recommendation System Statistics")
        st.write("📊 **Overview:** Analyze recommendation system coverage, data distribution, and cold-start challenges.")
        st.caption("Coverage metrics show how well the system can make recommendations. Cold-start analysis identifies users/movies with insufficient data.")
        
        try:
            ratings_df = pd.read_csv("ratings_data.csv")
            
            # Coverage statistics
            st.markdown("### 📊 Data Coverage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Ratings", f"{len(ratings_df):,}")
                st.metric("Unique Users", f"{ratings_df['user_id'].nunique():,}")
            
            with col2:
                st.metric("Unique Movies", f"{ratings_df['movie_id'].nunique():,}")
                avg_ratings_per_user = len(ratings_df) / ratings_df['user_id'].nunique()
                st.metric("Avg Ratings/User", f"{avg_ratings_per_user:.1f}")
            
            with col3:
                sparsity = 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['movie_id'].nunique()))
                st.metric("Matrix Sparsity", f"{sparsity:.2%}")
                st.caption("Lower is better (more dense)")
            
            # Distribution plots
            st.markdown("### 📈 Rating Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Ratings per user
                user_rating_counts = ratings_df["user_id"].value_counts()
                
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hist(user_rating_counts, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
                ax.axvline(user_rating_counts.median(), color="red", linestyle="--", linewidth=2, label=f"Median: {user_rating_counts.median():.0f}")
                ax.axvline(user_rating_counts.mean(), color="orange", linestyle="--", linewidth=2, label=f"Mean: {user_rating_counts.mean():.1f}")
                ax.set_xlabel("Number of Ratings per User")
                ax.set_ylabel("Frequency (User Count)")
                ax.set_title("Distribution of Ratings per User")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Ratings per movie
                movie_rating_counts = ratings_df["movie_id"].value_counts()
                
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hist(movie_rating_counts, bins=30, edgecolor="black", alpha=0.7, color="crimson")
                ax.axvline(movie_rating_counts.median(), color="blue", linestyle="--", linewidth=2, label=f"Median: {movie_rating_counts.median():.0f}")
                ax.axvline(movie_rating_counts.mean(), color="green", linestyle="--", linewidth=2, label=f"Mean: {movie_rating_counts.mean():.1f}")
                ax.set_xlabel("Number of Ratings per Movie")
                ax.set_ylabel("Frequency (Movie Count)")
                ax.set_title("Distribution of Ratings per Movie")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Cold start analysis
            st.markdown("### ❄️ Cold Start Analysis")
            
            cold_start_threshold = st.slider("Minimum ratings threshold", 1, 10, 5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                cold_users = (user_rating_counts < cold_start_threshold).sum()
                cold_users_pct = cold_users / len(user_rating_counts) * 100
                st.metric(
                    f"Users with < {cold_start_threshold} ratings",
                    f"{cold_users:,}",
                    delta=f"{cold_users_pct:.1f}% of users"
                )
            
            with col2:
                cold_movies = (movie_rating_counts < cold_start_threshold).sum()
                cold_movies_pct = cold_movies / len(movie_rating_counts) * 100
                st.metric(
                    f"Movies with < {cold_start_threshold} ratings",
                    f"{cold_movies:,}",
                    delta=f"{cold_movies_pct:.1f}% of movies"
                )
            
            # Genre distribution (if movies data available)
            movies_data = load_movies_data()
            if movies_data is not None and "genre_primary" in movies_data.columns:
                st.markdown("### 🎭 Genre Distribution in Recommendations")
                
                # Merge ratings with movies to get genres
                ratings_with_genre = ratings_df.merge(
                    movies_data[["movie_id", "genre_primary"]],
                    on="movie_id",
                    how="left"
                )
                
                genre_counts = ratings_with_genre["genre_primary"].value_counts().head(15)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                genre_counts.plot(kind="barh", ax=ax, color="teal", alpha=0.7)
                ax.set_xlabel("Number of Ratings")
                ax.set_ylabel("Genre")
                ax.set_title("Top 15 Genres by Rating Count")
                plt.tight_layout()
                st.pyplot(fig)
        
        except FileNotFoundError:
            st.warning("Ratings data not found. Please ensure ratings_data.csv is saved from your notebook.")

# =========================
# CLUSTER INSIGHTS PAGE
# =========================

def show_cluster_insights():
    st.markdown('<div class="main-header">📊 Customer Segmentation Insights</div>', unsafe_allow_html=True)
    
    st.write("""
    **Purpose:** Understand different customer segments, their characteristics, and business value.
    Segmentation uses unsupervised machine learning (K-Means clustering) to group similar customers together.
    """)
    
    # Load data
    user_features, churn_scored, cluster_personas = load_user_data()
    
    if cluster_personas is None or churn_scored is None:
        st.error("❌ Cluster data not loaded. Please ensure data files are available.")
        return
    
    # Overview metrics
    st.markdown("### 📈 Segmentation Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clusters", len(cluster_personas))
    with col2:
        st.metric("Total Users", cluster_personas["users_in_cluster"].sum())
    with col3:
        avg_churn = cluster_personas["churn_rate"].mean()
        st.metric("Avg Churn Rate", f"{avg_churn:.1%}")
    with col4:
        total_risk = cluster_personas["total_expected_value_loss"].sum()
        st.metric("Total Value at Risk", f"${total_risk:,.0f}")
    
    st.divider()
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Cluster Personas",
        "🔥 CLV vs Churn Heatmap",
        "🕸️ Radar Plots",
        "📊 Cluster Comparison"
    ])
    
    # Tab 1: Cluster Personas Table
    with tab1:
        st.markdown("### 📋 Cluster Personas")
        st.write("📊 **Overview:** Comprehensive table showing key metrics for each customer segment.")
        st.caption("Each cluster represents a group of customers with similar behaviors. Use this to identify high-value segments and those at risk.")
        
        # Format the personas table for display
        display_personas = cluster_personas.copy()
        display_personas["churn_rate"] = display_personas["churn_rate"].apply(lambda x: f"{x:.1%}")
        display_personas["avg_clv"] = display_personas["avg_clv"].apply(lambda x: f"${x:,.0f}")
        display_personas["total_expected_value_loss"] = display_personas["total_expected_value_loss"].apply(lambda x: f"${x:,.0f}")
        display_personas["avg_monthly_spend"] = display_personas["avg_monthly_spend"].apply(lambda x: f"${x:.2f}")
        display_personas["avg_total_watch_hours"] = display_personas["avg_total_watch_hours"].apply(lambda x: f"{x:.1f}h")
        display_personas["avg_completion_ratio"] = display_personas["avg_completion_ratio"].apply(lambda x: f"{x:.1%}")
        display_personas["avg_search_frequency"] = display_personas["avg_search_frequency"].apply(lambda x: f"{x:.3f}")
        display_personas["avg_review_rating"] = display_personas["avg_review_rating"].apply(lambda x: f"{x:.2f}")
        display_personas["avg_sentiment_score"] = display_personas["avg_sentiment_score"].apply(lambda x: f"{x:.2f}")
        display_personas["avg_kids_share"] = display_personas["avg_kids_share"].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_personas, use_container_width=True, hide_index=True)
        
        # Show detailed persona summaries
        st.markdown("### 💬 Persona Narratives")
        for _, row in cluster_personas.iterrows():
            with st.expander(f"Cluster {row['cluster']} - {row['users_in_cluster']} users"):
                st.write(row["persona_summary"])
    
    # Tab 2: CLV vs Churn Heatmap
    with tab2:
        st.markdown("### 🔥 CLV vs Churn Risk Matrix (Value at Risk)")
        st.write("🎯 **Overview:** Identify which customer segments represent the highest business risk.")
        st.caption("The heatmap shows total expected revenue loss (CLV × Churn Probability). Focus retention efforts on high-value, high-risk customers.")
        
        # Create the matrix from churn_scored data
        if "clv_band" in churn_scored.columns and "churn_risk_band" in churn_scored.columns:
            clv_churn_matrix = (
                churn_scored
                .groupby(["clv_band", "churn_risk_band"])["expected_value_loss"]
                .sum()
                .unstack("churn_risk_band", fill_value=0)
            )
            
            # Ensure consistent ordering
            band_order = ["low", "medium", "high"]
            clv_churn_matrix = clv_churn_matrix.reindex(
                index=[b for b in band_order if b in clv_churn_matrix.index],
                columns=[b for b in band_order if b in clv_churn_matrix.columns],
                fill_value=0
            )
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                clv_churn_matrix,
                annot=True,
                fmt=".0f",
                cmap="Reds",
                cbar_kws={"label": "Total Expected Value Loss ($)"},
                ax=ax
            )
            ax.set_xlabel("Churn Risk Band", fontsize=12)
            ax.set_ylabel("CLV Band", fontsize=12)
            ax.set_title("Value at Risk by CLV and Churn Risk", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show insights
            st.markdown("### 💡 Key Insights")
            high_risk_high_clv = clv_churn_matrix.loc["high", "high"] if "high" in clv_churn_matrix.index and "high" in clv_churn_matrix.columns else 0
            total_risk = clv_churn_matrix.sum().sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "High CLV / High Churn Risk",
                    f"${high_risk_high_clv:,.0f}",
                    delta=f"{(high_risk_high_clv/total_risk*100):.1f}% of total risk" if total_risk > 0 else "0%"
                )
            with col2:
                st.metric("Total Value at Risk", f"${total_risk:,.0f}")
        else:
            st.warning("CLV and churn risk bands not found in data.")
    
    # Tab 3: Radar Plots
    with tab3:
        st.markdown("### 🕸️ Cluster Profile Comparison")
        st.write("📐 **Overview:** Visual comparison of cluster characteristics using radar/spider charts.")
        st.caption("All metrics are normalized (0-1 scale) for fair comparison. Select up to 3 clusters to compare their profiles side-by-side.")
        
        # Select metrics for radar plot
        radar_metrics = [
            "avg_monthly_spend",
            "avg_total_watch_hours",
            "avg_completion_ratio",
            "avg_search_frequency",
            "avg_review_rating",
            "avg_sentiment_score",
            "avg_kids_share",
        ]
        
        # Get available metrics
        available_metrics = [m for m in radar_metrics if m in cluster_personas.columns]
        
        if len(available_metrics) < 3:
            st.warning("Not enough metrics available for radar plot.")
        else:
            # Normalize metrics to 0-1 scale
            radar_data = cluster_personas[available_metrics].copy()
            for col in available_metrics:
                col_min = radar_data[col].min()
                col_max = radar_data[col].max()
                if col_max > col_min:
                    radar_data[col] = (radar_data[col] - col_min) / (col_max - col_min)
                else:
                    radar_data[col] = 0.5
            
            # Select clusters to compare
            selected_clusters = st.multiselect(
                "Select clusters to compare (max 3)",
                options=cluster_personas["cluster"].tolist(),
                default=cluster_personas["cluster"].tolist()[:min(3, len(cluster_personas))],
                max_selections=3
            )
            
            if selected_clusters:
                from math import pi
                
                # Number of variables
                N = len(available_metrics)
                angles = [n / float(N) * 2 * pi for n in range(N)]
                angles += angles[:1]
                
                # Create radar plot
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
                
                colors = plt.cm.Set2(range(len(selected_clusters)))
                
                for idx, cluster_id in enumerate(selected_clusters):
                    cluster_idx = cluster_personas[cluster_personas["cluster"] == cluster_id].index[0]
                    values = radar_data.loc[cluster_idx, available_metrics].tolist()
                    values += values[:1]
                    
                    ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {cluster_id}", color=colors[idx])
                    ax.fill(angles, values, alpha=0.15, color=colors[idx])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([m.replace("avg_", "").replace("_", " ").title() for m in available_metrics], fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_title("Cluster Profiles: Normalized Feature Comparison", size=14, fontweight="bold", pad=20)
                ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
                ax.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Please select at least one cluster to display.")
    
    # Tab 4: Cluster Comparison
    with tab4:
        st.markdown("### 📊 Cluster Size vs Value at Risk")
        st.write("📈 **Overview:** Compare cluster sizes, value at risk, and churn rates across all segments.")
        st.caption("Dual-axis charts show both the number of customers and the financial risk in each segment.")
        
        # Dual-axis bar chart
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        clusters = cluster_personas["cluster"].values
        users_counts = cluster_personas["users_in_cluster"].values
        value_loss = cluster_personas["total_expected_value_loss"].values
        
        # Left axis: user count
        color1 = "steelblue"
        ax1.set_xlabel("Cluster", fontsize=12)
        ax1.set_ylabel("Users in Cluster", color=color1, fontsize=12)
        bars1 = ax1.bar(clusters - 0.2, users_counts, 0.4, label="Users", color=color1, alpha=0.7)
        ax1.tick_params(axis="y", labelcolor=color1)
        
        # Right axis: value at risk
        ax2 = ax1.twinx()
        color2 = "crimson"
        ax2.set_ylabel("Total Expected Value Loss ($)", color=color2, fontsize=12)
        bars2 = ax2.bar(clusters + 0.2, value_loss, 0.4, label="Value at Risk", color=color2, alpha=0.7)
        ax2.tick_params(axis="y", labelcolor=color2)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f"{int(height)}", ha="center", va="bottom", fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f"${int(height):,}", ha="center", va="bottom", fontsize=8, color=color2)
        
        plt.title("Cluster Size vs Total Expected Value Loss", fontsize=14, fontweight="bold")
        ax1.set_xticks(clusters)
        ax1.set_xticklabels([f"Cluster {c}" for c in clusters])
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Churn rate comparison
        st.markdown("### 📉 Churn Rate by Cluster")
        fig2, ax = plt.subplots(figsize=(10, 5))
        churn_rates = cluster_personas["churn_rate"].values * 100
        bars = ax.bar(clusters, churn_rates, color="coral", alpha=0.7)
        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Churn Rate (%)", fontsize=12)
        ax.set_title("Churn Rate by Customer Segment", fontsize=14, fontweight="bold")
        ax.set_xticks(clusters)
        ax.set_xticklabels([f"Cluster {c}" for c in clusters])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig2)

# =========================
# USER LOOKUP DASHBOARD
# =========================

def show_user_lookup():
    st.markdown('<div class="main-header">🔍 User Lookup Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    user_features, churn_scored, cluster_personas = load_user_data()
    
    if user_features is None or churn_scored is None:
        st.error("❌ Data not loaded. Please ensure data files are available.")
        st.info("💡 Tip: Run your notebook cells to generate and save the required CSV files.")
        return
    
    # Sidebar: User selection
    st.sidebar.header("🔎 User Selection")
    
    # Get list of available users
    available_users = sorted(churn_scored["user_id"].unique().tolist())
    
    # User input method: dropdown or text input
    input_method = st.sidebar.radio("Select user by:", ["Dropdown", "Text Input"])
    
    if input_method == "Dropdown":
        selected_user = st.sidebar.selectbox(
            "Choose a user ID:",
            available_users,
            index=0 if available_users else None
        )
    else:
        user_input = st.sidebar.text_input("Enter user ID:", value="")
        if user_input in available_users:
            selected_user = user_input
        elif user_input:
            st.sidebar.error(f"User '{user_input}' not found in dataset.")
            selected_user = None
        else:
            selected_user = None
    
    if selected_user is None:
        st.info("👆 Please select or enter a user ID from the sidebar.")
        return
    
    # Get user data
    user_churn_data = churn_scored[churn_scored["user_id"] == selected_user]
    user_feature_data = user_features[user_features["user_id"] == selected_user]
    
    if user_churn_data.empty:
        st.error(f"❌ User '{selected_user}' not found in churn data.")
        return
    
    # Extract key metrics
    churn_prob = user_churn_data["churn_proba"].iloc[0]
    clv = user_churn_data["clv_baseline"].iloc[0]
    expected_loss = user_churn_data["expected_value_loss"].iloc[0]
    churn_risk_band = user_churn_data["churn_risk_band"].iloc[0] if "churn_risk_band" in user_churn_data.columns else "N/A"
    clv_band = user_churn_data["clv_band"].iloc[0] if "clv_band" in user_churn_data.columns else "N/A"
    priority_segment = user_churn_data["priority_segment"].iloc[0] if "priority_segment" in user_churn_data.columns else "Other"
    actual_churn = user_churn_data["churn"].iloc[0] if "churn" in user_churn_data.columns else None
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_prob:.1%}",
            delta=f"{churn_prob - 0.5:.1%}" if churn_prob > 0.5 else None,
            delta_color="inverse"
        )
        st.caption(f"Risk Band: {churn_risk_band}")
    
    with col2:
        st.metric(
            "Customer Lifetime Value",
            f"${clv:,.0f}",
            delta=None
        )
        st.caption(f"CLV Band: {clv_band}")
    
    with col3:
        st.metric(
            "Expected Value Loss",
            f"${expected_loss:,.0f}",
            delta=None
        )
        st.caption("CLV × Churn Prob")
    
    with col4:
        priority_color = "🔴" if priority_segment == "High CLV / High Churn" else "🟢"
        st.metric(
            "Priority Segment",
            f"{priority_color} {priority_segment}",
            delta=None
        )
        if actual_churn is not None:
            churn_status = "✅ Retained" if actual_churn == 0 else "❌ Churned"
            st.caption(churn_status)
    
    st.divider()
    
    # Tabs for detailed views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 User Profile", "🧠 SHAP Explanation", "🎬 Recommendations", "📈 Feature Details"])
    
    # Tab 1: User Profile
    with tab1:
        st.subheader("User Profile & Characteristics")
        st.write("📋 **Overview:** Comprehensive view of user behavior, engagement, and subscription details.")
        st.caption("This profile includes tenure, spending patterns, viewing habits, and content preferences.")
        
        if not user_feature_data.empty:
            # Key features to display
            profile_features = [
                "tenure_days", "months_active", "monthly_spend", "plan_tier",
                "total_watch_hours", "avg_session_minutes", "completion_ratio",
                "device_diversity", "search_frequency", "rec_ctr",
                "avg_review_rating", "sentiment_score_avg", "complaint_count",
                "genre_diversity", "kids_content_share"
            ]
            
            available_features = [f for f in profile_features if f in user_feature_data.columns]
            
            if available_features:
                profile_df = user_feature_data[["user_id"] + available_features].T
                profile_df.columns = ["Value"]
                profile_df = profile_df.iloc[1:]  # remove user_id row
                profile_df.index.name = "Feature"
                profile_df = profile_df.reset_index()
                profile_df["Feature"] = profile_df["Feature"].str.replace("_", " ").str.title()
                
                st.dataframe(profile_df, use_container_width=True, hide_index=True)
            else:
                st.info("Feature data not available for this user.")
        else:
            st.info("User profile data not found.")
    
    # Tab 2: SHAP Explanation
    with tab2:
        st.subheader("Why did the model predict this churn probability?")
        st.write("🧠 **Overview:** SHAP (SHapley Additive exPlanations) values show which features push the churn prediction up or down.")
        st.caption("Red bars increase churn risk, green bars decrease it. Longer bars = stronger impact.")
        
        shap_explainer = load_shap_explainer()
        feature_cols = load_feature_columns()
        processed_features = load_processed_features()
        
        if shap_explainer is None:
            st.warning("⚠️ SHAP explainer not loaded. Please save it from your notebook.")
            st.info("💡 To enable SHAP explanations, save your explainer using:\n```python\nimport pickle\nwith open('shap_explainer.pkl', 'wb') as f:\n    pickle.dump(shap_explainer, f)\n```")
        elif feature_cols is None:
            st.warning("⚠️ Feature columns not found. Please save them from your notebook.")
        elif processed_features is None:
            st.warning("⚠️ Processed feature matrix not found. Please save it from your notebook.")
            st.info("""
            **How to fix this:**
            
            In your notebook, add this code to save the processed features:
            
            ```python
            # Save the processed feature matrix with user IDs
            churn_features_with_ids = churn_features.copy()
            churn_features_with_ids['user_id'] = user_activity_churn['user_id'].values
            churn_features_with_ids.to_csv('churn_features_processed.csv', index=False)
            ```
            """)
        else:
            # Get user's processed features
            user_processed = processed_features[processed_features["user_id"] == selected_user]
            
            if user_processed.empty:
                st.warning(f"⚠️ User {selected_user} not found in processed features. This user may not have been in the training/test set.")
                return
            
            try:
                import shap
                
                # Get user's feature values in the correct order (exclude user_id and churn columns)
                exclude_cols = ['user_id', 'churn']
                available_feature_cols = [col for col in feature_cols if col in user_processed.columns and col not in exclude_cols]
                
                X_user = user_processed[available_feature_cols].iloc[0:1]
                
                # Compute SHAP values
                shap_values = shap_explainer.shap_values(X_user)
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # Binary classification - use positive class (churn=1)
                    shap_values_display = shap_values[1][0]
                    expected_value = shap_explainer.expected_value[1] if isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value
                else:
                    shap_values_display = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                    expected_value = shap_explainer.expected_value
                
                # Create DataFrame of feature contributions
                contributions_df = pd.DataFrame({
                    'Feature': available_feature_cols,
                    'Value': X_user.iloc[0].values,
                    'SHAP Value': shap_values_display
                })
                contributions_df['Abs SHAP'] = contributions_df['SHAP Value'].abs()
                contributions_df = contributions_df.sort_values('Abs SHAP', ascending=False)
                
                # Display top contributing features
                st.markdown("### 🎯 Top Features Influencing Churn Prediction")
                
                top_n = st.slider("Number of top features to display", 5, 20, 10, key="shap_top_n")
                top_features = contributions_df.head(top_n)
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
                colors = ['red' if x > 0 else 'green' for x in top_features['SHAP Value']]
                ax.barh(range(len(top_features)), top_features['SHAP Value'], color=colors, alpha=0.7)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels([f.replace('_', ' ').title() for f in top_features['Feature']])
                ax.set_xlabel('SHAP Value (Impact on Churn Prediction)')
                ax.set_title('Feature Contributions to Churn Prediction')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("🔴 Red bars increase churn probability | 🟢 Green bars decrease churn probability")
                
                # Display detailed table
                st.markdown("### 📊 Detailed Feature Contributions")
                display_df = top_features[['Feature', 'Value', 'SHAP Value']].copy()
                display_df['Feature'] = display_df['Feature'].str.replace('_', ' ').str.title()
                display_df['SHAP Value'] = display_df['SHAP Value'].round(4)
                display_df['Impact'] = display_df['SHAP Value'].apply(
                    lambda x: f"{'↑' if x > 0 else '↓'} {'Increases' if x > 0 else 'Decreases'} churn risk"
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Show base value
                st.info(f"📌 **Base prediction (average):** {expected_value:.4f}")
                st.info(f"📌 **Final prediction for this user:** {churn_prob:.4f}")
                
            except Exception as e:
                st.error(f"❌ Error computing SHAP values: {str(e)}")
                st.info("This might happen if the feature columns don't match the model's expected features.")
    
    # Tab 3: Recommendations
    with tab3:
        st.subheader("🎬 Personalized Movie Recommendations")
        st.write("🎯 **Overview:** AI-powered movie suggestions based on collaborative filtering - what similar users enjoyed.")
        st.caption("Recommendations are filtered to exclude movies the user has already rated.")
        
        recommender_data = load_recommender()
        movies_data = load_movies_data()
        
        if recommender_data is None:
            st.warning("⚠️ Recommendation model not loaded.")
            st.info("""
            **To enable recommendations:**
            
            Make sure you have saved the recommendation model and ratings data from your notebook.
            The following files are needed:
            - `recommender_model.pkl`
            - `ratings_data.csv`
            """)
        else:
            model = recommender_data["model"]
            trainset = recommender_data["trainset"]
            all_movies = recommender_data["all_movies"]
            ratings_df = recommender_data["ratings_df"]
            
            # Check if user has ratings
            user_ratings = ratings_df[ratings_df["user_id"] == selected_user]
            
            if user_ratings.empty:
                st.warning(f"⚠️ User {selected_user} has no ratings in the system. Cannot generate recommendations.")
                st.info("This user is a cold-start user. Consider using popularity-based or content-based recommendations.")
            else:
                st.success(f"✅ User has {len(user_ratings)} ratings in the system")
                
                # Recommendation settings
                col1, col2 = st.columns(2)
                with col1:
                    n_recs = st.slider("Number of recommendations", 5, 20, 10, key="n_recs")
                with col2:
                    min_rating = st.slider("Minimum predicted rating", 1.0, 5.0, 3.5, 0.1, key="min_rating")
                
                if st.button("🎬 Get Recommendations", type="primary"):
                    with st.spinner("Generating recommendations..."):
                        try:
                            # Get movies user has already rated
                            try:
                                inner_uid = trainset.to_inner_uid(selected_user)
                                user_items = trainset.ur[inner_uid]
                                seen_movies = {trainset.to_raw_iid(iid) for iid, _ in user_items}
                            except ValueError:
                                seen_movies = set(user_ratings["movie_id"].unique())
                            
                            # Get candidate movies (not yet rated)
                            candidate_movies = [mid for mid in all_movies if mid not in seen_movies]
                            
                            if not candidate_movies:
                                st.warning("User has rated all available movies!")
                            else:
                                # Predict ratings for all candidates
                                predictions = []
                                for movie_id in candidate_movies:
                                    pred = model.predict(selected_user, movie_id, verbose=False)
                                    if pred.est >= min_rating:
                                        predictions.append((movie_id, pred.est))
                                
                                # Sort by predicted rating
                                predictions.sort(key=lambda x: x[1], reverse=True)
                                top_predictions = predictions[:n_recs]
                                
                                if not top_predictions:
                                    st.warning(f"No movies found with predicted rating >= {min_rating:.1f}")
                                else:
                                    # Create recommendations dataframe
                                    recs_df = pd.DataFrame(top_predictions, columns=["movie_id", "predicted_rating"])
                                    
                                    # Merge with movie metadata if available
                                    if movies_data is not None:
                                        recs_df = recs_df.merge(
                                            movies_data[["movie_id", "title", "genre_primary", "rating", "language"]],
                                            on="movie_id",
                                            how="left"
                                        )
                                        recs_df.rename(columns={"rating": "avg_rating"}, inplace=True)
                                    
                                    # Display recommendations
                                    st.markdown(f"### 🎯 Top {len(recs_df)} Recommendations")
                                    
                                    # Format the display
                                    display_recs = recs_df.copy()
                                    display_recs["predicted_rating"] = display_recs["predicted_rating"].round(2)
                                    if "avg_rating" in display_recs.columns:
                                        display_recs["avg_rating"] = display_recs["avg_rating"].round(2)
                                    
                                    st.dataframe(display_recs, use_container_width=True, hide_index=True)
                                    
                                    # Show genre distribution if available
                                    if "genre_primary" in recs_df.columns:
                                        st.markdown("### 📊 Genre Distribution")
                                        genre_counts = recs_df["genre_primary"].value_counts()
                                        
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        genre_counts.plot(kind="barh", ax=ax, color="steelblue")
                                        ax.set_xlabel("Number of Recommendations")
                                        ax.set_ylabel("Genre")
                                        ax.set_title("Recommended Movies by Genre")
                                        plt.tight_layout()
                                        st.pyplot(fig)
                        
                        except Exception as e:
                            st.error(f"❌ Error generating recommendations: {str(e)}")
                            st.info("This might happen if the model or data format is incompatible.")
    
    # Tab 4: Feature Details
    with tab4:
        st.subheader("Detailed Feature Breakdown")
        st.write("📈 **Overview:** Complete list of all user features and their values used in the analysis.")
        st.caption("These features feed into the churn prediction and recommendation models.")
        
        if not user_feature_data.empty:
            # Show all available features
            all_features = user_feature_data.select_dtypes(include=[np.number]).columns.tolist()
            all_features = [f for f in all_features if f not in ["user_id"]]
            
            if all_features:
                feature_values = user_feature_data[all_features].iloc[0]
                feature_df = pd.DataFrame({
                    "Feature": [f.replace("_", " ").title() for f in all_features],
                    "Value": feature_values.values
                })
                feature_df = feature_df.sort_values("Feature")
                
                st.dataframe(feature_df, use_container_width=True, hide_index=True)
            else:
                st.info("No numeric features available.")
        else:
            st.info("Feature details not available.")

# =========================
# MAIN APP
# =========================

def main():
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["About", "User Lookup Dashboard", "Cluster Insights", "Model Performance"],
        index=0
    )
    
    if page == "About":
        st.title("About This Platform")
        st.markdown("""
        ### 🎯 Customer Intelligence Platform
        
        This comprehensive analytics platform helps businesses understand customer behavior, predict churn, 
        and deliver personalized experiences using advanced machine learning techniques.
        
        ---
        
        ### 📊 Platform Features
        
        #### 1. User Lookup Dashboard
        Deep-dive into individual customer profiles with:
        - **Churn Risk Assessment**: Predict which customers are likely to leave
        - **Customer Lifetime Value (CLV)**: Understand each customer's long-term value
        - **SHAP Explanations**: See exactly why the model predicts a customer will churn
        - **Personalized Recommendations**: Generate movie suggestions tailored to each user
        
        #### 2. Cluster Insights
        Understand customer segments with:
        - **Cluster Personas**: Detailed profiles of each customer segment
        - **Value at Risk Heatmap**: Identify high-value customers at risk of churning
        - **Radar Plots**: Visual comparison of segment characteristics
        - **Segment Analysis**: Compare size, churn rates, and value across segments
        
        #### 3. Model Performance
        Evaluate and understand model quality with:
        - **Model Comparison**: Compare multiple ML algorithms side-by-side
        - **ROC & PR Curves**: Visualize model discrimination ability
        - **Feature Importance**: Identify which factors drive churn predictions
        - **Recommendation Stats**: Analyze recommendation system coverage and quality
        
        ---
        
        ### 🛠️ Technology Stack
        - **Frontend**: Streamlit
        - **ML Models**: scikit-learn, LightGBM, XGBoost
        - **Explainability**: SHAP (SHapley Additive exPlanations)
        - **Recommendations**: Surprise (collaborative filtering)
        - **Visualization**: Matplotlib, Seaborn
        
        ---
        
        ### 📈 Business Impact
        This platform enables data-driven decisions to:
        - Reduce customer churn through early intervention
        - Maximize customer lifetime value
        - Personalize user experiences at scale
        - Optimize marketing and retention strategies
        """)
    elif page == "User Lookup Dashboard":
        show_user_lookup()
    elif page == "Cluster Insights":
        show_cluster_insights()
    elif page == "Model Performance":
        show_model_performance()

if __name__ == "__main__":
    main()

