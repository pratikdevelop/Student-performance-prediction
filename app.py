import random
import pandas as pd

# Constants for the synthetic data
num_students = 1000  # Generating 500 student records (adjust as needed)

# Define possible values for each attribute
genders = ['0', '1']  # '0' = Male, '1' = Female
ethnicities = ['0', '1', '2']  # '0' = General, '1' = OBC, '2' = SC
par_ed = ['1', '2', '3']  # '1' = Low, '2' = Medium, '3' = High education level
study_time = ['10', '15', '20', '25', '30']  # Weekly study hours (in quotes)
absences_options = ['0', '5', '10', '15', '20', '25', '30']  # Absence days
tutoring = ['0', '1']  # '0' = No, '1' = Yes
par_support = ['1', '2', '3']  # '1' = Low, '2' = Medium, '3' = High parental support
extracurricular = ['0', '1']  # '0' = No, '1' = Yes
sports = ['0', '1']  # '0' = No, '1' = Yes
music = ['0', '1']  # '0' = No, '1' = Yes
volunteering = ['0', '1']  # '0' = No, '1' = Yes
gpa_range = ['1.0', '2.0', '3.0', '4.0']  # GPA range
grade_classes = ['1.0', '2.0', '3.0', '4.0']  # Grade classes

# Initialize an empty list for storing data rows
data = []

# Generate synthetic data for `num_students` students
for student_id in range(5000, 5000 + num_students):
    age = random.randint(15, 18)  # Random age between 15 and 18
    gender = random.choice(genders)  # Randomly select gender
    ethnicity = random.choice(ethnicities)  # Randomly select ethnicity
    parental_education = random.choice(par_ed)  # Randomly select parental education level
    study_time_weekly = random.choice(study_time)  # Randomly select study time (hours per week)
    absences = random.choice(absences_options)  # Randomly select absences (days)
    tutoring = random.choice(tutoring)  # Randomly select if tutoring is attended
    parental_support = random.choice(par_support)  # Randomly select parental support level
    extracurricular = random.choice(extracurricular)  # Randomly select extracurricular activities
    sports = random.choice(sports)  # Randomly select if involved in sports
    music = random.choice(music)  # Randomly select if involved in music
    volunteering = random.choice(volunteering)  # Randomly select if involved in volunteering
    gpa = round(random.uniform(1.0, 4.0), 2)  # Random GPA between 1.0 and 4.0
    grade_class = random.choice(grade_classes)  # Randomly assign grade class (1-4)

    # Create a data row for the student
    data.append([student_id, age, gender, ethnicity, parental_education, study_time_weekly, absences, tutoring, parental_support, extracurricular, sports, music, volunteering, gpa, grade_class])

# Convert the data into a DataFrame
columns = ['StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA', 'GradeClass']
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv('synthetic_student_data_extended.csv', index=False)

# Print the first few rows of the DataFrame to verify
print(df.head())
