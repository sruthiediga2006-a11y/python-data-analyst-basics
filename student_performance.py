import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = "student_performance.csv"

df = pd.read_csv(FILE_NAME)

print("=" * 60)
print("STUDENT PERFORMANCE ANALYSIS")
print("=" * 60)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

for column in ["Student", "Subject", "Result"]:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.title()
    )

df["Marks"] = pd.to_numeric(
    df["Marks"],
    errors="coerce"
)

df["Attendance"] = pd.to_numeric(
    df["Attendance"],
    errors="coerce"
)

df = df.dropna(
    subset=["Student", "Subject"]
)

df["Marks"] = df["Marks"].fillna(
    df["Marks"].median()
)

df["Attendance"] = df["Attendance"].fillna(
    df["Attendance"].median()
)

df["Result"] = df["Result"].fillna("Unknown")

print("\nDataset shape after cleaning:", df.shape)

average_marks = df["Marks"].mean()
highest_marks = df["Marks"].max()
lowest_marks = df["Marks"].min()

print("\n--- Marks Analysis ---")
print(f"Average Marks : {average_marks:.2f}")
print(f"Highest Marks : {highest_marks}")
print(f"Lowest Marks  : {lowest_marks}")

top_students = (
    df.groupby("Student")["Marks"]
    .mean()
    .nlargest(5)
    .round(2)
)

print("\n--- Top 5 Students ---")
print(top_students)

pass_percentage = (
    (df["Result"] == "Pass").mean() * 100
)

print("\n--- Result Analysis ---")
print(f"Pass Percentage: {pass_percentage:.2f}%")
print(f"Not Passed / Other: {100 - pass_percentage:.2f}%")

subject_average = (
    df.groupby("Subject")["Marks"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

print("\n--- Subject-wise Average Marks ---")
print(subject_average)

plt.figure(figsize=(8, 5))
subject_average.plot(kind="bar", color="steelblue")
plt.title("Average Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Marks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("subject_averages.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(
    df["Marks"],
    bins=10,
    color="seagreen",
    edgecolor="black"
)
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(
    df["Attendance"],
    df["Marks"],
    alpha=0.6,
    color="darkorange"
)
plt.title("Attendance vs Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("attendance_vs_marks.png")
plt.show()

correlation = df["Attendance"].corr(
    df["Marks"]
)

low_attendance_marks = df[
    df["Attendance"] < 75
]["Marks"].mean()

high_attendance_marks = df[
    df["Attendance"] >= 75
]["Marks"].mean()

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print(
    f"1. Overall average marks: "
    f"{average_marks:.1f}"
)

print(
    f"2. Marks range from "
    f"{lowest_marks} to {highest_marks}."
)

print(
    f"3. Strongest subject: "
    f"{subject_average.idxmax()} "
    f"({subject_average.max():.1f})"
)

print(
    f"4. Weakest subject: "
    f"{subject_average.idxmin()} "
    f"({subject_average.min():.1f})"
)

print(
    f"5. Pass percentage: "
    f"{pass_percentage:.1f}%"
)

relationship = (
    "positive"
    if correlation > 0
    else "negative"
)

print(
    f"6. Attendance and marks have a "
    f"{relationship} correlation of "
    f"{correlation:.2f}."
)

print(
    f"7. Students with 75%+ attendance "
    f"average {high_attendance_marks:.1f} marks."
)

print(
    f"8. Students with below 75% attendance "
    f"average {low_attendance_marks:.1f} marks."
)

print("\nAnalysis completed successfully!")