# Student Segmentation and Personalized Course Recommendation System for EduPro

## 1. Abstract

Online learning platforms serve learners with different interests, experience levels, learning behaviors, and course preferences. Providing the same recommendations to every learner may not effectively address these differences.

This project develops a **Student Segmentation and Personalized Course Recommendation System for EduPro**. The system analyzes learner, course, and transaction data to understand learner behavior and identify groups of learners with similar characteristics.

The project uses data preprocessing, feature engineering, and **K-Means clustering** to segment learners based on their enrollment activity, spending behavior, course preferences, learning depth, and engagement patterns. Different cluster sizes were evaluated using inertia and silhouette scores. The analysis selected **2 clusters**, with a silhouette score of **0.6056**.

The resulting learner segments were **Focused / Low-Activity Learners** and **Advanced / Deep Learners**. The system also generates course recommendations using course popularity and rating information.

An interactive **Streamlit dashboard** was developed to allow users to explore learner profiles, compare learner segments, view behavioral characteristics, examine clustering results, and access course recommendations.

The completed system demonstrates how learner data and machine learning can be used to support learner segmentation and personalized course discovery.

---

# 2. Introduction

The growth of online education platforms has resulted in large amounts of learner and course-related data. This data contains useful information about learner interests, enrollment behavior, spending patterns, course preferences, and engagement.

However, learners do not behave in the same way. Some learners may enroll in several courses across different categories, while others may focus on a smaller number of courses. Some learners may prefer advanced courses, while others may primarily select beginner or intermediate courses.

A one-size-fits-all recommendation approach may therefore provide courses that are not equally relevant to all learners.

This project addresses this challenge by developing a learner segmentation and personalized recommendation system for the EduPro online learning platform. Learner-level behavioral features are created from transaction and course data, and machine learning is used to group learners with similar characteristics.

The resulting learner segments provide a structured way to understand different types of learners. A recommendation component then provides course suggestions, while an interactive Streamlit dashboard makes the analysis accessible through visualizations and learner-level exploration.

---

# 3. Problem Statement

Online learning platforms often provide recommendations without fully considering differences in learner behavior and preferences.

The main problems addressed in this project are:

* Lack of structured learner segmentation.
* Difficulty understanding learner enrollment behavior.
* Limited use of learner-level behavioral information.
* Generic course recommendations.
* Difficulty comparing different learner groups.
* Lack of an interactive interface for exploring learner analytics.

The project aims to solve these problems by creating a data-driven learner segmentation and course recommendation system.

---

# 4. Objectives

The main objectives of the project are:

1. Analyze learner enrollment and transaction behavior.
2. Combine learner, course, and transaction information.
3. Create meaningful learner-level behavioral features.
4. Segment learners using K-Means clustering.
5. Evaluate different numbers of clusters using clustering metrics.
6. Identify meaningful learner segments.
7. Develop a course recommendation component.
8. Build an interactive Streamlit dashboard.
9. Provide an easy-to-use learner profile explorer.
10. Support data-driven understanding of learner preferences and engagement.

---

# 5. Dataset Description

The project uses three primary datasets from the EduPro platform:

## 5.1 Users Dataset

The Users dataset contains information about learners.

| Column   | Description               |
| -------- | ------------------------- |
| UserID   | Unique learner identifier |
| UserName | Learner name              |
| Age      | Learner age               |
| Gender   | Learner gender            |
| Email    | Learner email             |

The dataset contains **3,000 learner records**.

## 5.2 Courses Dataset

The Courses dataset contains information about available courses.

| Column         | Description              |
| -------------- | ------------------------ |
| CourseID       | Unique course identifier |
| CourseName     | Course name              |
| CourseCategory | Course category          |
| CourseType     | Course type              |
| CourseLevel    | Course difficulty level  |
| CoursePrice    | Course price             |
| CourseDuration | Course duration          |
| CourseRating   | Course rating            |

The dataset contains **60 courses**.

## 5.3 Transactions Dataset

The Transactions dataset records learner course transactions.

| Column          | Description                   |
| --------------- | ----------------------------- |
| TransactionID   | Unique transaction identifier |
| UserID          | Learner identifier            |
| CourseID        | Course identifier             |
| TransactionDate | Date of transaction           |
| Amount          | Transaction amount            |
| PaymentMethod   | Payment method                |
| TeacherID       | Teacher identifier            |

The dataset contains **10,000 transactions**.

---

# 6. Technology Stack

The following technologies were used:

* **Python** — Main programming language
* **Pandas** — Data loading, cleaning, transformation, and analysis
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine learning and clustering
* **Matplotlib** — Analytical visualizations
* **Plotly** — Interactive dashboard visualizations
* **Streamlit** — Interactive web dashboard
* **Jupyter / VS Code** — Development and analysis environment
* **CSV** — Dataset and analytical output format

---

# 7. System Architecture

The overall system follows a data processing and machine learning pipeline.

```text
Users Dataset
      |
      |
Courses Dataset -----> Data Cleaning & Merging
      |                       |
      |                       v
Transactions Dataset    Feature Engineering
                              |
                              v
                     Learner-Level Profiles
                              |
                              v
                     Feature Standardization
                              |
                              v
                       K-Means Clustering
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Learner Segments          Cluster Evaluation
                 |
                 v
       Course Recommendation
                 |
                 v
          Streamlit Dashboard
```

---

# 8. Data Preprocessing

Before performing machine learning, the datasets were cleaned and prepared.

The main preprocessing steps included:

### 8.1 Loading the datasets

The three CSV files were loaded using Pandas.

### 8.2 Data type conversion

Important columns such as transaction dates, numerical amounts, ratings, and course attributes were converted into appropriate data types.

### 8.3 Transaction and course merging

Transaction records were merged with course information using `CourseID`.

This allowed transaction activity to be analyzed together with course category, level, duration, and rating.

### 8.4 Learner information integration

Learner information was incorporated using `UserID`.

This allowed transactions to be aggregated at the learner level.

### 8.5 Handling learner-level information

Transaction-level records were converted into learner-level profiles so that each learner could be represented by behavioral characteristics rather than individual transactions.

---

# 9. Feature Engineering

Feature engineering was an important part of the project.

The following learner-level features were created.

| Feature                   | Description                           |
| ------------------------- | ------------------------------------- |
| TotalCoursesEnrolled      | Number of unique courses enrolled     |
| TotalTransactions         | Number of transactions                |
| TotalSpending             | Total learner spending                |
| AverageSpending           | Average transaction spending          |
| AverageCourseRating       | Average rating of enrolled courses    |
| DiversityScore            | Diversity of course categories        |
| FirstEnrollment           | Date of first enrollment              |
| LastEnrollment            | Date of latest enrollment             |
| AverageCourseDuration     | Average duration of enrolled courses  |
| PreferredCourseCategory   | Most frequently selected category     |
| PreferredCourseLevel      | Most frequently selected course level |
| AdvancedCourses           | Number of advanced courses            |
| AdvancedCourseRatio       | Ratio of advanced courses             |
| ActiveMonths              | Number of active learning months      |
| EnrollmentFrequency       | Enrollment activity over time         |
| LearningDepthIndex        | Indicator of learning depth           |
| AverageCoursesPerCategory | Average courses across categories     |
| EngagementScore           | Combined learner engagement measure   |

These features provide a broader representation of learner behavior.

---

# 10. Feature Standardization

The numerical features used for clustering were standardized before applying K-Means.

Standardization helps place variables with different numerical ranges on a comparable scale.

For example, total spending may have values much larger than a rating score. Without scaling, variables with larger numerical values could have a disproportionate influence on the clustering algorithm.

The project therefore used feature scaling before clustering.

---

# 11. Learner Segmentation Using K-Means

K-Means clustering was selected to group learners with similar behavioral characteristics.

The algorithm works by:

1. Selecting a number of clusters.
2. Assigning learners to cluster centers.
3. Calculating distances between learners and cluster centers.
4. Updating cluster centers.
5. Repeating the process until the clusters stabilize.

Several values of K were tested, from **2 through 8**.

---

# 12. Cluster Evaluation

Two important measurements were used:

## 12.1 Inertia

Inertia measures the within-cluster sum of squared distances.

Lower inertia generally indicates that observations are closer to their cluster centers.

The tested values were:

|  K |  Inertia | Silhouette |
| -: | -------: | ---------: |
|  2 | 16000.76 |     0.6056 |
|  3 | 13154.74 |     0.3120 |
|  4 | 10709.90 |     0.3647 |
|  5 |  8677.09 |     0.4014 |
|  6 |  7742.03 |     0.4210 |
|  7 |  7335.08 |     0.3870 |
|  8 |  6347.65 |     0.4267 |

## 12.2 Silhouette Score

The silhouette score measures how well observations fit within their assigned clusters compared with other clusters.

The highest silhouette score obtained during the tested range was:

**0.6056 for K = 2**

Based on the implemented selection procedure, **2 clusters** were selected for the final learner segmentation.

---

# 13. Learner Segments

The final clustering produced two learner segments.

| Segment                         | Number of Learners |
| ------------------------------- | -----------------: |
| Focused / Low-Activity Learners |              2,546 |
| Advanced / Deep Learners        |                454 |
| **Total**                       |          **3,000** |

## 13.1 Focused / Low-Activity Learners

This segment contains the larger group of learners.

The segment represents learners whose behavioral features resulted in a comparatively lower-activity profile within the clustering model.

Possible characteristics include relatively lower enrollment activity and lower learning depth compared with the other identified segment.

The segment label is a descriptive interpretation assigned after examining the cluster characteristics.

## 13.2 Advanced / Deep Learners

This segment contains 454 learners.

The segment represents learners whose behavioral characteristics resulted in a comparatively stronger learning-depth profile.

Features such as advanced-course activity and engagement-related measures contributed to the segmentation.

The segment name is a descriptive interpretation of the observed learner profile.

---

# 14. Course Recommendation System

After learner segmentation, a recommendation component was developed.

The recommendation process uses available course information and course popularity/rating information to identify relevant courses.

The system considers:

* Course enrollment popularity
* Course ratings
* Learner information
* Learner segment
* Course characteristics

The system generates a set of course recommendations for learners.

The recommendation output contains course-level information that can be displayed through the Streamlit dashboard.

---

# 15. Recommendation Approach

The recommendation component follows a cluster-aware approach.

The general process is:

```text
Learner
   |
   v
Learner Profile
   |
   v
Learner Segment
   |
   v
Course Candidates
   |
   v
Popularity + Rating Information
   |
   v
Recommended Courses
```

This approach provides a structured recommendation process rather than displaying a completely random list of courses.

The recommendation system can also be extended in the future with more advanced content-based or collaborative filtering techniques.

---

# 16. Streamlit Dashboard

An interactive dashboard was developed using Streamlit.

The dashboard provides several sections.

## 16.1 Key Statistics

The dashboard displays:

* Total learners
* Total courses
* Total transactions
* Number of learner segments

The current dataset contains:

* **3,000 learners**
* **60 courses**
* **10,000 transactions**
* **2 learner segments**

## 16.2 Learner Segment Distribution

A visualization displays the number of learners in each identified segment.

This allows users to compare the size of the learner groups.

## 16.3 Segment Comparison

The dashboard displays the cluster summary so that characteristics of the learner groups can be compared.

## 16.4 Learner Profile Explorer

A learner can be selected from the dashboard.

The selected learner's information includes:

* Learner ID
* Number of courses enrolled
* Total spending
* Learner segment
* Preferred course category
* Preferred course level
* Average course rating

## 16.5 Learner Behavior

A chart displays selected behavioral metrics for the chosen learner.

These include:

* Courses enrolled
* Transactions
* Advanced courses
* Active months

## 16.6 Personalized Recommendations

The dashboard displays recommended courses for the selected learner.

## 16.7 Clustering Evaluation

The dashboard also provides the clustering evaluation output and silhouette score visualization.

---

# 17. Project Outputs

The analysis generated the following files:

```text
data/output/
│
├── cluster_evaluation.csv
├── cluster_summary.csv
├── elbow_curve.png
├── learner_profiles.csv
├── merged_transaction_data.csv
├── recommendations.csv
└── silhouette_scores.png
```

These files provide reusable outputs for further analysis and dashboard development.

---

# 18. Results

The completed analysis produced the following major results:

* **3,000 learners** were analyzed.
* **60 courses** were included.
* **10,000 transactions** were processed.
* Learner-level behavioral profiles were successfully created.
* K-Means clustering was evaluated for K values from 2 to 8.
* The implemented selection procedure selected **K = 2**.
* The highest observed silhouette score was **0.6056** at K = 2.
* Two learner segments were created.
* The larger segment contains **2,546 learners**.
* The second segment contains **454 learners**.
* Course recommendations were successfully generated.
* An interactive Streamlit dashboard was successfully developed and tested.

---

# 19. Benefits of the System

The proposed system provides several benefits.

### For learners

* Easier discovery of relevant courses.
* Recommendations based on learner information.
* Better understanding of personal learning behavior.

### For the EduPro platform

* Structured understanding of learner groups.
* Ability to analyze learner engagement patterns.
* Support for targeted recommendation strategies.
* Interactive exploration of learner data.
* Reusable analytical datasets.

### For analysts

* Learner-level behavioral features.
* Machine-learning-based segmentation.
* Visual evaluation of clustering.
* Exportable analytical outputs.

---

# 20. Limitations

The current system also has some limitations.

1. The recommendation system is primarily based on course popularity and rating information rather than a full collaborative filtering model.

2. The project uses historical transaction data, so learner behavior may change over time.

3. The number of clusters depends on the selected features and clustering evaluation method.

4. The learner segment names are descriptive interpretations of the resulting clusters.

5. The project does not currently use real-time learner activity.

6. Recommendation quality was not evaluated using a production-scale A/B test.

7. The current dataset may not contain every factor that influences learner course selection.

---

# 21. Future Scope

The system can be improved in several ways.

### 21.1 Advanced Recommendation Algorithms

Future versions can implement:

* Collaborative filtering
* Content-based filtering
* Hybrid recommendation systems
* Matrix factorization
* Neural recommendation models

### 21.2 Real-Time Recommendations

The system could be connected to live learner activity so recommendations can update based on recent enrollments and interactions.

### 21.3 Improved Segmentation

Additional algorithms such as:

* Hierarchical clustering
* DBSCAN
* Gaussian Mixture Models

could be compared with K-Means.

### 21.4 Recommendation Evaluation

Future work can measure:

* Precision
* Recall
* F1-score
* Click-through rate
* Course enrollment rate
* Completion rate
* Engagement lift

### 21.5 Personalized Learning Paths

Instead of recommending individual courses, the system could generate complete learning paths based on learner goals and current skill levels.

---

# 22. Conclusion

This project developed a **Student Segmentation and Personalized Course Recommendation System for EduPro** using learner, course, and transaction data.

The project transformed transaction-level data into learner-level behavioral profiles using feature engineering. K-Means clustering was then applied to identify groups of learners with similar characteristics.

After evaluating multiple cluster sizes, the implemented analysis selected **two learner segments**, with a silhouette score of **0.6056** for K = 2. The resulting groups were described as **Focused / Low-Activity Learners** and **Advanced / Deep Learners**.

A course recommendation component was also developed using course popularity and rating information. Finally, the complete analysis was presented through an interactive Streamlit dashboard containing learner statistics, segment visualizations, learner profile exploration, recommendations, and clustering evaluation.

Overall, the project demonstrates how machine learning and data analytics can be combined to understand learner behavior and support personalized course discovery on an online learning platform.

---

# 23. Project Folder Structure

The completed project is organized as follows:

```text
C:\EduPro_Project
│
├── data
│   ├── Users.csv
│   ├── Courses.csv
│   ├── Transactions.csv
│   │
│   └── output
│       ├── cluster_evaluation.csv
│       ├── cluster_summary.csv
│       ├── elbow_curve.png
│       ├── learner_profiles.csv
│       ├── merged_transaction_data.csv
│       ├── recommendations.csv
│       └── silhouette_scores.png
│
├── notebooks
│   └── analysis.py
│
├── dashboard
│   └── app.py
│
└── report
    └── EduPro_Project_Report.md
```

---

# 24. Tools and Technologies Summary

| Category         | Technology         |
| ---------------- | ------------------ |
| Programming      | Python             |
| Data Processing  | Pandas, NumPy      |
| Machine Learning | Scikit-learn       |
| Clustering       | K-Means            |
| Visualization    | Matplotlib, Plotly |
| Dashboard        | Streamlit          |
| Development      | VS Code            |
| Data Storage     | CSV                |

---

# 25. Final Project Summary

**Project Title:** Student Segmentation and Personalized Course Recommendation System for EduPro

**Learners Analyzed:** 3,000

**Courses:** 60

**Transactions:** 10,000

**Final Clusters:** 2

**Best Observed Silhouette Score:** 0.6056

**Segments:**

* Focused / Low-Activity Learners
* Advanced / Deep Learners

**Main Deliverables:**

* Data analysis pipeline
* Learner segmentation model
* Course recommendation system
* Analytical output datasets
* Streamlit dashboard
* Project report
