import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

DATA_FOLDER = "../data"
OUTPUT_FOLDER = "../data/output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING EDUPRO DATASETS")
print("=" * 60)

users = pd.read_csv(
    os.path.join(DATA_FOLDER, "Users.csv")
)

courses = pd.read_csv(
    os.path.join(DATA_FOLDER, "Courses.csv")
)

transactions = pd.read_csv(
    os.path.join(DATA_FOLDER, "Transactions.csv")
)

print("\nUsers:", users.shape)
print("Courses:", courses.shape)
print("Transactions:", transactions.shape)


# ============================================================
# 3. CLEAN DATA
# ============================================================

users.columns = users.columns.str.strip()
courses.columns = courses.columns.str.strip()
transactions.columns = transactions.columns.str.strip()

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    errors="coerce"
)

transactions["Amount"] = pd.to_numeric(
    transactions["Amount"],
    errors="coerce"
)

courses["CourseRating"] = pd.to_numeric(
    courses["CourseRating"],
    errors="coerce"
)

courses["CoursePrice"] = pd.to_numeric(
    courses["CoursePrice"],
    errors="coerce"
)

courses["CourseDuration"] = pd.to_numeric(
    courses["CourseDuration"],
    errors="coerce"
)


# ============================================================
# 4. MERGE TRANSACTIONS + COURSE INFORMATION
# ============================================================

print("\nMerging transaction and course data...")

data = transactions.merge(
    courses,
    on="CourseID",
    how="left"
)

# Add user information
data = data.merge(
    users[["UserID", "Age", "Gender"]],
    on="UserID",
    how="left"
)

print("Merged dataset:", data.shape)


# ============================================================
# 5. CREATE LEARNER PROFILES
# ============================================================

print("\nCreating learner profiles...")

profiles = (
    data.groupby("UserID")
    .agg(
        TotalCoursesEnrolled=("CourseID", "nunique"),
        TotalTransactions=("TransactionID", "count"),
        TotalSpending=("Amount", "sum"),
        AverageSpending=("Amount", "mean"),
        AverageCourseRating=("CourseRating", "mean"),
        DiversityScore=("CourseCategory", "nunique"),
        FirstEnrollment=("TransactionDate", "min"),
        LastEnrollment=("TransactionDate", "max"),
        AverageCourseDuration=("CourseDuration", "mean")
    )
    .reset_index()
)


# ============================================================
# 6. PREFERRED CATEGORY
# ============================================================

category_counts = (
    data.groupby(["UserID", "CourseCategory"])
    .size()
    .reset_index(name="Count")
)

preferred_category = (
    category_counts
    .sort_values("Count", ascending=False)
    .drop_duplicates("UserID")
    [["UserID", "CourseCategory"]]
    .rename(
        columns={
            "CourseCategory": "PreferredCourseCategory"
        }
    )
)

profiles = profiles.merge(
    preferred_category,
    on="UserID",
    how="left"
)


# ============================================================
# 7. PREFERRED LEVEL
# ============================================================

level_counts = (
    data.groupby(["UserID", "CourseLevel"])
    .size()
    .reset_index(name="Count")
)

preferred_level = (
    level_counts
    .sort_values("Count", ascending=False)
    .drop_duplicates("UserID")
    [["UserID", "CourseLevel"]]
    .rename(
        columns={
            "CourseLevel": "PreferredCourseLevel"
        }
    )
)

profiles = profiles.merge(
    preferred_level,
    on="UserID",
    how="left"
)


# ============================================================
# 8. ADVANCED COURSE RATIO
# ============================================================

advanced = (
    data["CourseLevel"]
    .astype(str)
    .str.lower()
    .str.contains("advanced")
)

advanced_counts = (
    data.loc[advanced]
    .groupby("UserID")
    .size()
    .rename("AdvancedCourses")
)

profiles = profiles.merge(
    advanced_counts,
    on="UserID",
    how="left"
)

profiles["AdvancedCourses"] = (
    profiles["AdvancedCourses"]
    .fillna(0)
)

profiles["AdvancedCourseRatio"] = (
    profiles["AdvancedCourses"]
    / profiles["TotalCoursesEnrolled"]
)


# ============================================================
# 9. ACTIVE MONTHS
# ============================================================

profiles["ActiveMonths"] = (
    (
        profiles["LastEnrollment"]
        - profiles["FirstEnrollment"]
    ).dt.days / 30
)

profiles["ActiveMonths"] = (
    profiles["ActiveMonths"]
    .clip(lower=1)
)


# ============================================================
# 10. ENROLLMENT FREQUENCY
# ============================================================

profiles["EnrollmentFrequency"] = (
    profiles["TotalCoursesEnrolled"]
    / profiles["ActiveMonths"]
)


# ============================================================
# 11. LEARNING DEPTH INDEX
# ============================================================

profiles["LearningDepthIndex"] = (
    profiles["TotalCoursesEnrolled"]
    * (
        1 + profiles["AdvancedCourseRatio"]
    )
)


# ============================================================
# 12. AVERAGE COURSES PER CATEGORY
# ============================================================

profiles["AverageCoursesPerCategory"] = (
    profiles["TotalCoursesEnrolled"]
    / profiles["DiversityScore"].replace(0, np.nan)
)

profiles["AverageCoursesPerCategory"] = (
    profiles["AverageCoursesPerCategory"]
    .fillna(profiles["TotalCoursesEnrolled"])
)


# ============================================================
# 13. ENGAGEMENT SCORE
# ============================================================

profiles["EngagementScore"] = (
    profiles["TotalCoursesEnrolled"]
    + profiles["EnrollmentFrequency"]
    + profiles["LearningDepthIndex"]
)


# ============================================================
# 14. CLEAN NUMERIC DATA
# ============================================================

numeric_features = [
    "TotalCoursesEnrolled",
    "TotalTransactions",
    "TotalSpending",
    "AverageSpending",
    "AverageCourseRating",
    "DiversityScore",
    "AdvancedCourseRatio",
    "ActiveMonths",
    "EnrollmentFrequency",
    "LearningDepthIndex",
    "AverageCoursesPerCategory",
    "EngagementScore"
]

profiles[numeric_features] = (
    profiles[numeric_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# 15. PREPARE CLUSTERING DATA
# ============================================================

print("\nPreparing clustering data...")

X = profiles[numeric_features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 16. TEST K VALUES
# ============================================================

print("\nTesting K-Means clusters...")

cluster_results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    cluster_results.append({
        "K": k,
        "Inertia": model.inertia_,
        "SilhouetteScore": silhouette
    })

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette = {silhouette:.4f}"
    )


cluster_results = pd.DataFrame(
    cluster_results
)


# ============================================================
# 17. SELECT BEST K
# ============================================================

best_k = int(
    cluster_results.loc[
        cluster_results["SilhouetteScore"].idxmax(),
        "K"
    ]
)

print("\nBest K:", best_k)


# ============================================================
# 18. FINAL K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

profiles["Cluster"] = (
    kmeans.fit_predict(X_scaled)
)


# ============================================================
# 19. CREATE SEGMENT NAMES
# ============================================================

cluster_summary = (
    profiles
    .groupby("Cluster")[numeric_features]
    .mean()
)

overall_engagement = (
    profiles["EngagementScore"].mean()
)

overall_depth = (
    profiles["LearningDepthIndex"].mean()
)

overall_diversity = (
    profiles["DiversityScore"].mean()
)

segment_names = {}

for cluster in cluster_summary.index:

    engagement = cluster_summary.loc[
        cluster,
        "EngagementScore"
    ]

    depth = cluster_summary.loc[
        cluster,
        "LearningDepthIndex"
    ]

    diversity = cluster_summary.loc[
        cluster,
        "DiversityScore"
    ]

    if depth > overall_depth * 1.25:

        name = "Advanced / Deep Learners"

    elif diversity > overall_diversity * 1.25:

        name = "Exploratory / Diverse Learners"

    elif engagement > overall_engagement * 1.25:

        name = "Active / General Learners"

    else:

        name = "Focused / Low-Activity Learners"

    segment_names[cluster] = name


profiles["Segment"] = (
    profiles["Cluster"]
    .map(segment_names)
)


# ============================================================
# 20. CREATE COURSE POPULARITY
# ============================================================

course_popularity = (
    data.groupby("CourseID")
    .agg(
        EnrollmentCount=("UserID", "count"),
        AverageRating=("CourseRating", "mean")
    )
    .reset_index()
)

course_popularity["RecommendationScore"] = (
    course_popularity["EnrollmentCount"] * 0.4
    + course_popularity["AverageRating"].fillna(0) * 0.6
)

course_popularity = (
    course_popularity
    .sort_values(
        "RecommendationScore",
        ascending=False
    )
)


# ============================================================
# 21. CREATE TOP 5 RECOMMENDATIONS
# ============================================================

print("\nCreating recommendations...")

top_courses = course_popularity.head(5)

recommendations = []

for user_id in profiles["UserID"]:

    for rank, course_id in enumerate(
        top_courses["CourseID"],
        start=1
    ):

        recommendations.append({
            "UserID": user_id,
            "RecommendationRank": rank,
            "RecommendedCourseID": course_id
        })


recommendations = pd.DataFrame(
    recommendations
)


# ============================================================
# 22. SAVE LEARNER PROFILES
# ============================================================

profiles.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "learner_profiles.csv"
    ),
    index=False
)


# ============================================================
# 23. SAVE CLUSTER EVALUATION
# ============================================================

cluster_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "cluster_evaluation.csv"
    ),
    index=False
)


# ============================================================
# 24. SAVE CLUSTER SUMMARY
# ============================================================

cluster_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "cluster_summary.csv"
    )
)


# ============================================================
# 25. SAVE RECOMMENDATIONS
# ============================================================

recommendations.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "recommendations.csv"
    ),
    index=False
)


# ============================================================
# 26. SAVE MERGED DATA
# ============================================================

data.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "merged_transaction_data.csv"
    ),
    index=False
)


# ============================================================
# 27. ELBOW CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cluster_results["K"],
    cluster_results["Inertia"],
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve")

plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "elbow_curve.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 28. SILHOUETTE CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cluster_results["K"],
    cluster_results["SilhouetteScore"],
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")

plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "silhouette_scores.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("EDUPRO ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nNumber of learners:", len(profiles))

print(
    "Number of courses:",
    courses["CourseID"].nunique()
)

print(
    "Number of transactions:",
    len(transactions)
)

print(
    "Best number of clusters:",
    best_k
)

print("\nLearner segments:")

print(
    profiles["Segment"]
    .value_counts()
)

print("\nOutput files:")

for file in os.listdir(OUTPUT_FOLDER):
    print("-", file)

print("\nAll analysis files are saved in:")

print(
    os.path.abspath(OUTPUT_FOLDER)
)