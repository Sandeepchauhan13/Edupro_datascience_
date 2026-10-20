import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================
# EDUPro LEARNER ANALYTICS — INSIGHT-DRIVEN DASHBOARD
# Expected files in the same folder:
# courses.csv, teachers.csv, transactions.csv, users.csv
# =========================================================

# st.set_page_config
# It controls the browser tab title, tab icon,
# and page layout. Place it near the beginning of your Streamlit script, before other Streamlit display commands.
st.set_page_config(
    page_title="EduPro Learner Analytics",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduPro Learner Analytics Dashboard")
st.write(
    "Understand learner behavior, identify meaningful learner segments, "
    "and replace one-size-fits-all suggestions with personalized course recommendations."
)
# --------------------------- Data loading ---------------------------

@st.cache_data
# def is the Python keyword used to define a function.
# These lines load four CSV files into pandas DataFrames.
def load_data():
    courses = pd.read_csv("courses.csv")
    teachers = pd.read_csv("teachers.csv")
    transactions = pd.read_csv("transactions.csv")
    users = pd.read_csv("users.csv")

# Required  creates a Python dictionary containing sets. There are two concepts to understand here. A. Dictionary
# A dictionary stores information as key-value pairs. Just like , name is key and Sandy is value
# A set is a collection of unique values. 
    required = {
        "courses": {"CourseID", "CourseName", "CourseCategory", "CourseType",
                    "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"},
        "teachers": {"TeacherID", "TeacherName", "TeacherRating", "Expertise",
                     "YearsOfExperience"},
        "transactions": {"TransactionID", "UserID", "CourseID", "TransactionDate",
                         "Amount", "PaymentMethod", "TeacherID"},
        "users": {"UserID", "Age", "Gender"}
    }
    frames = {"courses": courses, "teachers": teachers,
              "transactions": transactions, "users": users}


# This loop goes through each key-value pair in the required dictionary.
# The .items() method provides each dictionary key and its corresponding value.
    for name, columns in required.items():

        missing = columns - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name}.csv is missing required columns: {sorted(missing)}")
# frames[name] retrieves the relevant DataFrame.
# .columns returns its column names.
# set(...) converts those column names into a set.
# columns - ... finds required column names that are absent from the DataFrame.
# The result is assigned to missing.
    for frame, key in [(courses, "CourseID"), (teachers, "TeacherID"),
                       (transactions, "TransactionID"), (users, "UserID")]:
        if frame[key].duplicated().any() and key != "TransactionID":
            # Repeated transaction IDs are not expected either; report duplicates below.
            pass

    transactions["TransactionDate"] = pd.to_datetime(
        transactions["TransactionDate"], errors="coerce", dayfirst=True
    )
    transactions["Amount"] = pd.to_numeric(transactions["Amount"], errors="coerce").fillna(0)
    courses["CoursePrice"] = pd.to_numeric(courses["CoursePrice"], errors="coerce").fillna(0)
    courses["CourseRating"] = pd.to_numeric(courses["CourseRating"], errors="coerce")
    teachers["TeacherRating"] = pd.to_numeric(teachers["TeacherRating"], errors="coerce")
    return courses, teachers, transactions, users
try:
    courses, teachers, transactions, users = load_data()

except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.info("Keep courses.csv, teachers.csv, transactions.csv and users.csv beside this Python file.")
    st.stop()

# Merge tables; keep all transactions and make missing relationships visible.
# This code combines your four datasets into one table,
# checks whether transaction data exists, and calculates a reference date for analyzing learner behavior.
# - data is the variable that will store the final combined DataFrame.
# - = assigns the result to that variable.
# - Parentheses allow you to write a long expression across multiple lines.
# - The dots at the beginning of each continued line are method calls on the preceding DataFrame.
# This style is called method chaining: performing several operations one after another 
# without creating a separate variable for every intermediate result.
# merge()
# Combines two DataFrames by matching values in one or more columns.
# on="CourseID"
# Specifies the column used to match the transactions with the courses.
# how="left"
# Keeps every row from the left DataFrame, which is transactions. Matching course details are added where available.
# validate="many_to_one"
# Checks that the right DataFrame (courses) has no duplicate CourseID values. Multiple transactions
# can refer to the same course, but each course ID must appear at most once in the courses table.

data = (
    transactions
    .merge(courses, on="CourseID", how="left", validate="many_to_one")
    .merge(teachers, on="TeacherID", how="left", suffixes=("", "_Teacher"),
           validate="many_to_one")
    .merge(users, on="UserID", how="left", suffixes=("", "_User"),
           validate="many_to_one")
)

if data.empty:
    st.warning("No transaction records were found.")
    st.stop()

valid_dates = data["TransactionDate"].dropna()
reference_date = (
    valid_dates.max() + pd.Timedelta(days=1)
    if not valid_dates.empty else pd.Timestamp.today().normalize()
)
    # --------------------------- Learner features ---------------------------
# groupby() groups rows with the same value in the UserID column.
# For example, all transactions belonging to U01 are grouped together, and all transactions belonging to U02 are grouped separately.
learner = (
    data.groupby("UserID", dropna=False)
    .agg(
        LastActivity=("TransactionDate", "max"),
        Frequency=("TransactionID", "nunique"),
        TotalSpend=("Amount", "sum"),
        UniqueCourses=("CourseID", "nunique"),
        UniqueCategories=("CourseCategory", "nunique"),
        AvgCourseRating=("CourseRating", "mean"),
        AvgTeacherRating=("TeacherRating", "mean"),
        Age=("Age", "first"),
        Gender=("Gender", "first")
    )
    .reset_index()
)

# Calculate how many days have passed since each learner's last activity.
# reference_date: The date from which you measure recency.
# learner["LastActivity"]: The date when the learner last made a transaction or had recorded activity.
# reference_date - learner["LastActivity"]: Subtracts the last activity date from the reference date.
# .dt.days: Extracts the difference in whole days.
# learner["RecencyDays"]: Stores the result in a new column.

learner["RecencyDays"] = (reference_date - learner["LastActivity"]).dt.days
learner["RecencyDays"] = learner["RecencyDays"].fillna(
    learner["RecencyDays"].max() if learner["RecencyDays"].notna().any() else 0
).clip(lower=0)

paid_share = (
    data.assign(IsPaid=data["Amount"].gt(0))
    .groupby("UserID")["IsPaid"].mean()
    .rename("PaidCourseShare")
)
learner = learner.merge(paid_share, on="UserID", how="left", validate="one_to_one")
learner["PaidCourseShare"] = learner["PaidCourseShare"].fillna(0)
learner["CategoryDiversity"] = (
    learner["UniqueCategories"] /
    learner["UniqueCourses"].replace(0, np.nan)
).fillna(0)

# --------------------------- Learner segmentation ---------------------------

segment_features = [
    "RecencyDays", "Frequency", "TotalSpend",
    "UniqueCourses", "UniqueCategories", "PaidCourseShare"
]
X = learner[segment_features].replace([np.inf, -np.inf], np.nan).fillna(0)

# KMeans needs at least as many rows as clusters.
# Tells K-Means to divide learners into two groups, called clusters.
# For example, one group might contain highly engaged learners, while another contains less engaged learners.
if len(learner) >= 2:
    scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    learner["Cluster"] = model.fit_predict(scaled)
    activity = learner.groupby("Cluster").agg(
        MeanFrequency=("Frequency", "mean"),
        MeanSpend=("TotalSpend", "mean"),
        MeanRecency=("RecencyDays", "mean")
    )
     # Higher engagement = more transactions/spend and lower recency.
    activity["EngagementIndex"] = (
        activity["MeanFrequency"].rank(pct=True) +
        activity["MeanSpend"].rank(pct=True) +
        (1 - activity["MeanRecency"].rank(pct=True))
    )
    active_cluster = activity["EngagementIndex"].idxmax()
    learner["Segment"] = np.where(
        learner["Cluster"].eq(active_cluster),
        "High engagement",
        "Occasional / low engagement"
    )
else:
    learner["Cluster"] = 0
    learner["Segment"] = "Insufficient data for segmentation"
    # --------------------------- Sidebar filters ---------------------------

st.sidebar.header("🎛️ Dashboard Filters")
segment_options = ["All"] + sorted(learner["Segment"].dropna().unique().tolist())
selected_segment = st.sidebar.selectbox("Learner segment", segment_options)

category_options = ["All"] + sorted(data["CourseCategory"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Course category", category_options)

filtered_learners = learner.copy()
if selected_segment != "All":
    filtered_learners = filtered_learners[filtered_learners["Segment"] == selected_segment]

filtered_data = data.copy()
if selected_category != "All":
    filtered_data = filtered_data[filtered_data["CourseCategory"] == selected_category]

# --------------------------- KPI row ---------------------------

total_learners = data["UserID"].nunique()
total_courses = courses["CourseID"].nunique()
total_transactions = data["TransactionID"].nunique()
total_revenue = data["Amount"].sum()
paid_transactions = int(data["Amount"].gt(0).sum())
free_transactions = int(data["Amount"].eq(0).sum())
paid_share_pct = (paid_transactions / total_transactions * 100) if total_transactions else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Learners", f"{total_learners:,}")
k2.metric("Courses", f"{total_courses:,}")
k3.metric("Transactions", f"{total_transactions:,}")
k4.metric("Total revenue", f"₹{total_revenue:,.0f}")
k5.metric("Paid transaction share", f"{paid_share_pct:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💡 Business Insights", "📊 Learner Behavior", "👥 Segmentation",
    "🎯 Recommendations", "📚 Course Analysis"
])
# --------------------------- Tab 1: insights ---------------------------

with tab1:
    st.header("Business Insights")
    st.caption("Insights below are calculated from the currently loaded CSV files.")

    category_summary = (
        data.groupby("CourseCategory", dropna=False)
        .agg(
            Learners=("UserID", "nunique"),
            Transactions=("TransactionID", "nunique"),
            Revenue=("Amount", "sum"),
            AvgCourseRating=("CourseRating", "mean")
        )
        .reset_index()
    )
    category_summary["RevenueSharePct"] = (
        category_summary["Revenue"] / total_revenue * 100 if total_revenue else 0
    )
    category_summary = category_summary.sort_values("Revenue", ascending=False)

    top_revenue = category_summary.iloc[0] if not category_summary.empty else None
    top_learners = category_summary.sort_values("Learners", ascending=False).iloc[0] if not category_summary.empty else None
    lowest_rating = category_summary.dropna(subset=["AvgCourseRating"]).sort_values("AvgCourseRating").head(1)
    monthly = (
        data.dropna(subset=["TransactionDate"])
        .assign(Month=lambda x: x["TransactionDate"].dt.to_period("M").astype(str))
        .groupby("Month")
        .agg(Transactions=("TransactionID", "nunique"), Revenue=("Amount", "sum"))
        .reset_index()
        .sort_values("Month")
    )
    peak_month = monthly.sort_values("Revenue", ascending=False).head(1)

    avg_transactions = learner["Frequency"].mean()
    median_transactions = learner["Frequency"].median()
    avg_spend = learner["TotalSpend"].mean()
    repeat_pct = (learner["Frequency"].gt(1).mean() * 100) if len(learner) else 0
    high_engagement_pct = (
        learner["Segment"].eq("High engagement").mean() * 100 if len(learner) else 0
    )

    a, b, c = st.columns(3)
    a.metric("Average transactions per learner", f"{avg_transactions:.2f}")
    b.metric("Learners with repeat transactions", f"{repeat_pct:.1f}%")
    c.metric("High-engagement segment", f"{high_engagement_pct:.1f}%")

    st.subheader("What the data says")
    insights = []
    insights.append(
        f"**Monetization:** {paid_transactions:,} of {total_transactions:,} transactions "
        f"({paid_share_pct:.1f}%) have a positive amount; {free_transactions:,} are zero-amount transactions."
    )
    if top_revenue is not None:
        insights.append(
            f"**Revenue leader:** **{top_revenue['CourseCategory']}** generated "
            f"₹{top_revenue['Revenue']:,.0f} ({top_revenue['RevenueSharePct']:.1f}% of total revenue)."
        )
    if top_learners is not None:
        insights.append(
            f"**Reach leader:** **{top_learners['CourseCategory']}** reached "
            f"{int(top_learners['Learners']):,} distinct learners."
        )
    if not lowest_rating.empty:
        row = lowest_rating.iloc[0]
        insights.append(
            f"**Quality opportunity:** **{row['CourseCategory']}** has the lowest category-average "
            f"course rating ({row['AvgCourseRating']:.2f}/5); investigate course feedback before scaling promotion."
        )
    if not peak_month.empty:
        row = peak_month.iloc[0]
        insights.append(
            f"**Best revenue month in the loaded data:** **{row['Month']}**, with ₹{row['Revenue']:,.0f}."
        )
    insights.append(
        f"**Engagement:** average learner transaction count is {avg_transactions:.2f}; "
        f"the median is {median_transactions:.0f}. Compare the mean and median to see whether activity is concentrated among frequent learners."
    )
    for item in insights:
        st.markdown(f"- {item}")

    st.subheader("Revenue by category")
    fig = px.bar(category_summary, x="CourseCategory", y="Revenue",
                 hover_data=["Learners", "Transactions", "RevenueSharePct"],
                 title="Which categories generate revenue?")
    fig.update_layout(xaxis_title="", yaxis_title="Revenue (₹)")
    st.plotly_chart(fig, use_container_width=True)

    columns = ["UserID", "Segment", "RecencyDays", "Frequency", "TotalSpend",
               "UniqueCourses", "UniqueCategories", "PaidCourseShare",
               "AvgCourseRating", "AvgTeacherRating"]
    st.dataframe(
        filtered_learners[columns].sort_values("Frequency", ascending=False),
        use_container_width=True
    )

# --------------------------- Tab 3: segmentation ---------------------------

with tab3:
    st.header("👥 Structured Learner Segmentation")
    st.write(
        "K-Means uses standardized recency, transaction frequency, spend, course breadth, "
        "category breadth and paid-transaction share. Segment names are assigned after comparing cluster behavior."
    )

    segment_summary = (
        learner.groupby("Segment")
        .agg(
            Learners=("UserID", "nunique"),
            AvgRecencyDays=("RecencyDays", "mean"),
            AvgTransactions=("Frequency", "mean"),
            AvgSpend=("TotalSpend", "mean"),
            AvgCourses=("UniqueCourses", "mean"),
            AvgCategories=("UniqueCategories", "mean"),
            AvgPaidShare=("PaidCourseShare", "mean")
        )
        .reset_index()
    )
    segment_summary["LearnerSharePct"] = segment_summary["Learners"] / max(len(learner), 1) * 100
    st.dataframe(segment_summary.round(2), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.pie(segment_summary, names="Segment", values="Learners",
                     title="Learner share by segment")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(segment_summary, x="Segment", y="AvgTransactions",
                     title="Average transactions by segment")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(learner, x="Frequency", y="TotalSpend", color="Segment",
                     size="UniqueCourses", hover_data=["UserID", "RecencyDays"],
                     title="Learner segments: activity vs spend")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------- Tab 4: recommendations ---------------------------

with tab4:
    st.header("🎯 Personalized Course Recommendations")
    st.write(
        "The ranking combines a learner's category and level preferences with course rating, "
        "teacher rating, and the learner's observed preference for paid vs free transactions."
    )

    learner_ids = sorted(learner["UserID"].dropna().unique().tolist())
    if not learner_ids:
        st.warning("No learners are available for recommendations.")
    else:
        selected_user = st.selectbox("Select learner", learner_ids)
        user_history = data[data["UserID"] == selected_user].copy()
        purchased_courses = set(user_history["CourseID"].dropna())

        category_pref = user_history["CourseCategory"].value_counts(normalize=True).to_dict()
        level_pref = user_history["CourseLevel"].value_counts(normalize=True).to_dict()
        paid_preference = float(user_history["Amount"].gt(0).mean()) if len(user_history) else 0

        # Teacher rating is linked to the course using the teacher observed on its transactions.
        teacher_by_course = (
            data.groupby("CourseID")["TeacherRating"].mean().to_dict()
        )

        recs = courses[~courses["CourseID"].isin(purchased_courses)].copy()
        recs["CategoryScore"] = recs["CourseCategory"].map(category_pref).fillna(0)
        recs["LevelScore"] = recs["CourseLevel"].map(level_pref).fillna(0)
        recs["CourseRatingScore"] = recs["CourseRating"].fillna(0).clip(0, 5) / 5
        recs["TeacherRatingScore"] = (
            recs["CourseID"].map(teacher_by_course).fillna(teachers["TeacherRating"].mean()).clip(0, 5) / 5
        )
        recs["PaidScore"] = np.where(recs["CoursePrice"].gt(0),
                                     paid_preference, 1 - paid_preference)
        
        # Weighted, explainable ranking; all components are scaled approximately 0-1.
        recs["RecommendationScore"] = (
        recs["CategoryScore"] * 0.40 +
        recs["LevelScore"] * 0.20 +
        recs["CourseRatingScore"] * 0.20 +
        recs["TeacherRatingScore"] * 0.10 +
        recs["PaidScore"] * 0.10
        )
        recs["RecommendationReason"] = np.select(
            [
                recs["CategoryScore"].gt(0) & recs["LevelScore"].gt(0),
                recs["CategoryScore"].gt(0),
                recs["LevelScore"].gt(0)
            ],
        [
                "Matches preferred category and level",
                "Matches preferred category",
                "Matches preferred course level"
            ],
        default="Discover a new category"
        )
        top_recs = recs.sort_values(
            ["RecommendationScore", "CourseRating"], ascending=[False, False]
        ).head(10)

        st.subheader(f"Top recommendations for {selected_user}")
        if top_recs.empty:
            st.info("This learner has already purchased every course in the catalogue.")
        else:
            st.dataframe(
                top_recs[["CourseID", "CourseName", "CourseCategory", "CourseLevel",
                          "CourseType", "CoursePrice", "CourseRating",
                          "RecommendationScore", "RecommendationReason"]].reset_index(drop=True),
                use_container_width=True
            )
            fig = px.bar(top_recs.sort_values("RecommendationScore"),
                         x="RecommendationScore", y="CourseName", orientation="h",
                         color="CourseCategory", title="Top recommendation scores")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Learner's previous transactions")
        history_columns = ["TransactionDate", "CourseID", "CourseName", "CourseCategory",
                           "CourseLevel", "Amount", "CourseRating"]
        st.dataframe(
            user_history[history_columns].sort_values("TransactionDate", ascending=False),
            use_container_width=True
        )

# --------------------------- Tab 5: course analysis ---------------------------

with tab5:
    st.header("📚 Course & Category Performance")
    course_data = filtered_data.copy()

    category_summary = (
        course_data.groupby("CourseCategory", dropna=False)
        .agg(
            Learners=("UserID", "nunique"),
            Transactions=("TransactionID", "nunique"),
            Revenue=("Amount", "sum"),
            AvgRating=("CourseRating", "mean")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(category_summary, x="CourseCategory", y="Learners",
                               title="Distinct learners by category"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(category_summary, x="CourseCategory", y="Revenue",
                               title="Revenue by category"), use_container_width=True)

    course_summary = (
        course_data.groupby(["CourseID", "CourseName", "CourseCategory"], dropna=False)
        .agg(
            Learners=("UserID", "nunique"),
            Transactions=("TransactionID", "nunique"),
            Revenue=("Amount", "sum"),
            AvgRating=("CourseRating", "mean")
        )
        .reset_index()
        .sort_values(["Revenue", "Learners"], ascending=False)
    )
    st.subheader("Course performance table")
    st.dataframe(course_summary, use_container_width=True)

# --------------------------- Data quality ---------------------------

with st.expander("Data checks"):
    checks = pd.DataFrame({
        "Check": [
            "Transactions with invalid/missing date",
            "Transactions with unmatched course",
            "Transactions with unmatched teacher",
            "Transactions with unmatched user",
            "Duplicate transaction IDs",
            "Missing course category after merge"
        ],
        "Count": [
            int(data["TransactionDate"].isna().sum()),
            int(data["CourseName"].isna().sum()),
            int(data["TeacherName"].isna().sum()),
            int(data["Age"].isna().sum()),
            int(transactions["TransactionID"].duplicated().sum()),
            int(data["CourseCategory"].isna().sum())
        ]
    })
    st.dataframe(checks, use_container_width=True)

st.markdown("---")
st.caption("EduPro Learner Analytics | Python • Pandas • Plotly • Scikit-learn • Streamlit")
