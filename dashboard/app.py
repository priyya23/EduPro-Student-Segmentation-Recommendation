import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPro Learner Analytics",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "output")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    profiles = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "learner_profiles.csv")
    )

    cluster_summary = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "cluster_summary.csv")
    )

    recommendations = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "recommendations.csv")
    )

    evaluation = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "cluster_evaluation.csv")
    )

    return profiles, cluster_summary, recommendations, evaluation


profiles, cluster_summary, recommendations, evaluation = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 EduPro Learner Segmentation & Course Recommendation System")

st.markdown(
    """
    This dashboard analyzes learner behavior, identifies learner segments,
    and provides personalized course recommendations.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Filters")

segments = sorted(profiles["Segment"].dropna().unique())

selected_segment = st.sidebar.multiselect(
    "Select Learner Segment",
    options=segments,
    default=segments
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_profiles = profiles[
    profiles["Segment"].isin(selected_segment)
]


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

st.subheader("📊 Key Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Learners",
        f"{len(profiles):,}"
    )

with col2:
    st.metric(
        "Total Courses",
        "60"
    )

with col3:
    st.metric(
        "Total Transactions",
        "10,000"
    )

with col4:
    st.metric(
        "Learner Segments",
        profiles["Cluster"].nunique()
    )


# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

st.subheader("👥 Learner Segment Distribution")

segment_counts = (
    filtered_profiles["Segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["Segment", "Learners"]

fig_segments = px.bar(
    segment_counts,
    x="Segment",
    y="Learners",
    title="Number of Learners in Each Segment",
    text="Learners"
)

fig_segments.update_layout(
    xaxis_title="Learner Segment",
    yaxis_title="Number of Learners"
)

st.plotly_chart(
    fig_segments,
    use_container_width=True
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

st.subheader("📋 Segment Comparison")

st.dataframe(
    cluster_summary,
    use_container_width=True
)


# ============================================================
# LEARNER PROFILE EXPLORER
# ============================================================

st.subheader("🔎 Learner Profile Explorer")

learner_ids = sorted(
    filtered_profiles["UserID"].dropna().unique()
)

selected_user = st.selectbox(
    "Select a Learner",
    learner_ids
)

learner = profiles[
    profiles["UserID"] == selected_user
].iloc[0]


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Learner ID",
        str(learner["UserID"])
    )

with col2:
    st.metric(
        "Courses Enrolled",
        int(learner["TotalCoursesEnrolled"])
    )

with col3:
    st.metric(
        "Total Spending",
        f"{learner['TotalSpending']:.2f}"
    )

with col4:
    st.metric(
        "Segment",
        learner["Segment"]
    )


st.write("### Learner Preferences")

pref_col1, pref_col2, pref_col3 = st.columns(3)

with pref_col1:
    st.write(
        "**Preferred Category:**",
        learner["PreferredCourseCategory"]
    )

with pref_col2:
    st.write(
        "**Preferred Level:**",
        learner["PreferredCourseLevel"]
    )

with pref_col3:
    st.write(
        "**Average Course Rating:**",
        round(learner["AverageCourseRating"], 2)
    )


# ============================================================
# LEARNER BEHAVIOR
# ============================================================

st.subheader("📈 Learner Behavior")

behavior_data = pd.DataFrame({
    "Metric": [
        "Courses Enrolled",
        "Transactions",
        "Advanced Courses",
        "Active Months"
    ],
    "Value": [
        learner["TotalCoursesEnrolled"],
        learner["TotalTransactions"],
        learner["AdvancedCourses"],
        learner["ActiveMonths"]
    ]
})

fig_behavior = px.bar(
    behavior_data,
    x="Metric",
    y="Value",
    title="Selected Learner Behavior"
)

st.plotly_chart(
    fig_behavior,
    use_container_width=True
)


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

st.subheader("🎯 Personalized Course Recommendations")

user_recommendations = recommendations[
    recommendations["UserID"] == selected_user
]

if len(user_recommendations) > 0:

    st.dataframe(
        user_recommendations,
        use_container_width=True
    )

else:

    st.info(
        "No recommendations available for this learner."
    )


# ============================================================
# CLUSTER QUALITY
# ============================================================

st.subheader("📐 Clustering Evaluation")

evaluation_display = evaluation.copy()

st.dataframe(
    evaluation_display,
    use_container_width=True
)


# ============================================================
# SILHOUETTE SCORE VISUALIZATION
# ============================================================

if "Silhouette" in evaluation.columns:

    fig_silhouette = px.line(
        evaluation,
        x="K",
        y="Silhouette",
        markers=True,
        title="Silhouette Score by Number of Clusters"
    )

    fig_silhouette.update_layout(
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score"
    )

    st.plotly_chart(
        fig_silhouette,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "EduPro Online Learning Platform | "
    "Student Segmentation & Personalized Course Recommendation System"
)