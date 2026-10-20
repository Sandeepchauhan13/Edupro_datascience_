    courses = pd.read_csv("courses.csv")
    teachers = pd.read_csv("teachers.csv")
    transactions = pd.read_csv("transactions.csv")
    edupro = pd.read_csv("edupro.csv")

    transactions["TransactionDate"] = pd.to_datetime(
        transactions["TransactionDate"],
        errors="coerce",
        dayfirst=True
    )

    return courses, teachers, transactions, edupro


courses, teachers, transactions, edupro = load_data()


# =========================================================
# MERGE DATA
# =========================================================

data = transactions.merge(
    courses,
    on="CourseID",
    how="left"
)

data = data.merge(
    teachers,
    on="TeacherID",
    how="left",
    suffixes=("", "_Teacher")
)


# =========================================================
# BASIC KPIs
# =========================================================

total_learners = data["UserID"].nunique()
total_courses = data["CourseID"].nunique()
total_transactions = data["TransactionID"].nunique()
total_revenue = data["Amount"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👨‍🎓 Learners",
    f"{total_learners:,}"
)

col2.metric(
    "📚 Courses",
    f"{total_courses:,}"
)

col3.metric(
    "🛒 Transactions",
    f"{total_transactions:,}"
)

col4.metric(
    "💰 Revenue",
    f"₹{total_revenue:,.0f}"
)


# =========================================================
# LEARNER BEHAVIOR FEATURES
# =========================================================

reference_date = data["TransactionDate"].max() + pd.Timedelta(days=1)

learner = (
    data.groupby("UserID")
    .agg(
        LastActivity=("TransactionDate", "max"),
        Frequency=("TransactionID", "count"),
        TotalSpend=("Amount", "sum"),
        UniqueCourses=("CourseID", "nunique"),
        UniqueCategories=("CourseCategory", "nunique"),
        AvgCourseRating=("CourseRating", "mean"),
        AvgTeacherRating=("TeacherRating", "mean")
    )
    .reset_index()
)

learner["RecencyDays"] = (
    reference_date - learner["LastActivity"]
).dt.days

learner["PaidCourseShare"] = (
    data.assign(Paid=data["Amount"] > 0)
    .groupby("UserID")["Paid"]
    .mean()
    .values
)

learner["CategoryDiversity"] = (
    learner["UniqueCategories"] /
    learner["UniqueCourses"]
)
