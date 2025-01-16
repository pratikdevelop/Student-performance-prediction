import pandas as pd
import shap
import sqlite3
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from model_training import load_arff_data, preprocess_synthetic_dataset, merge_datasets, train_combined_model, save_model, load_model  # Import from model_training.py

# Initialize Flask app
app = Flask(__name__)

# Configure file upload
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Define a function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_new_dataset(file_path):
    # Load the new dataset
    new_data = pd.read_csv(file_path)
    return new_data

def preprocess_student_data(file_path):
    student_data = pd.read_csv(file_path, sep=';')
    
    # Preprocessing categorical variables
    categorical_columns = [
        'sex', 'school', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'
    ]
    
    for column in categorical_columns:
        student_data[column] = student_data[column].map({'no': 0, 'yes': 1, 'M': 0, 'F': 1, 'GP': 0, 'MS': 1, 'U': 0, 'R': 1})
    
    # Drop rows with missing values
    student_data.dropna(inplace=True)
    
    # Select features and target variable (G3)
    features = student_data.drop(columns=['G3'])
    target = student_data['G3']
    
    return features, target


def generate_learning_path_and_feedback(student, predicted_grade):
    feedback = {
        'learning_path': [],
        'feedback': ''
    }
    
    if predicted_grade < 10:
        feedback['learning_path'].append("Needs improvement in core subjects. Focus on fundamental concepts.")
        feedback['feedback'] = "You need to work harder. Review the fundamentals."
    elif predicted_grade < 15:
        feedback['learning_path'].append("Average performance. Practice more problems for deeper understanding.")
        feedback['feedback'] = "You're doing okay. Try to engage more with practice problems."
    else:
        feedback['learning_path'].append("Excellent performance. Start exploring advanced topics.")
        feedback['feedback'] = "Excellent performance! Keep up the hard work."
    
    if student['studytime'] == 1:
        feedback['learning_path'].append("Consider increasing your weekly study time for better results.")
    elif student['studytime'] == 2:
        feedback['learning_path'].append("Adequate study time. Keep a steady pace.")
    elif student['studytime'] == 3:
        feedback['learning_path'].append("Good study habits. Try focusing on practice.")
    else:
        feedback['learning_path'].append("Excellent study time. Keep up the good work.")
    
    if student['failures'] > 0:
        feedback['learning_path'].append("You have failed in past classes. Review past lessons thoroughly.")
    
    return feedback

# Function for SHAP (Explainable AI) 
def explain_prediction(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], X)

# Function to create SQLite database for performance tracking
def create_database():
    conn = sqlite3.connect('student_performance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS student_performance (
                    student_id INTEGER PRIMARY KEY,
                    predicted_grade REAL,
                    actual_grade REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_performance(student_id, predicted_grade, actual_grade):
    conn = sqlite3.connect('student_performance.db')
    c = conn.cursor()
    c.execute("INSERT INTO student_performance (student_id, predicted_grade, actual_grade) VALUES (?, ?, ?)",
              (student_id, predicted_grade, actual_grade))
    conn.commit()
    conn.close()

def get_performance_history(student_id):
    conn = sqlite3.connect('student_performance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM student_performance WHERE student_id = ?", (student_id,))
    history = c.fetchall()
    conn.close()
    return history

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        sex = int(request.form['sex'])
        school = int(request.form['school'])
        address = int(request.form['address'])
        famsize = int(request.form['famsize'])
        Pstatus = int(request.form['Pstatus'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        Mjob = request.form['Mjob']
        Fjob = request.form['Fjob']
        reason = request.form['reason']
        guardian = request.form['guardian']
        age = int(request.form['age'])

        # Prepare the input data for prediction
        new_student = {
            'sex': sex,
            'school': school,
            'address': address,
            'famsize': famsize,
            'Pstatus': Pstatus,
            'studytime': studytime,
            'failures': failures,
            'Mjob': Mjob,
            'Fjob': Fjob,
            'reason': reason,
            'guardian': guardian,
            'age': age
        }

        # Convert new student data into a DataFrame
        new_student_df = pd.DataFrame([new_student])

        # Load and preprocess the existing data
        student_features, student_target = preprocess_student_data('/home/pc-25/Music/AI-platform/student+performance/student/student-por.csv')
        arff_features, arff_target = load_arff_data('/home/pc-25/Music/AI-platform/student+performance/student/CEE_DATA.arff')
        new_data = load_new_dataset('/home/pc-25/Music/AI-platform/student+performance/student/Student_performance_data _.csv')  # New dataset file
        synthetic_data = load_new_dataset('/home/pc-25/Music/AI-platform/synthetic_student_data_extended.csv')  # Your synthetic dataset

        # Preprocess the datasets
        new_features, new_target = preprocess_synthetic_dataset(new_data)
        synthetic_features, synthetic_target = preprocess_synthetic_dataset(synthetic_data)
        
        # Merge all datasets
        combined_features, combined_target = merge_datasets(student_features, student_target, arff_features, arff_target)
        combined_features_new, combined_target_new = merge_datasets(combined_features, combined_target, new_features, new_target)
        combined_features_all, combined_target_all = merge_datasets(combined_features_new, combined_target_new, synthetic_features, synthetic_target)

        # Define the categorical columns for encoding
        categorical_columns = ['Mjob', 'Fjob', 'reason', 'guardian']

        # Train the model on all combined datasets
        model = train_combined_model(combined_features_all, combined_target_all, categorical_columns)

        # Save the trained model
        save_model(model)

        # Ensure new student data matches expected columns (same order as training data)
        expected_columns = combined_features_all.columns.tolist()
        new_student_df = new_student_df.reindex(columns=expected_columns, fill_value=0)

        # Predict the student's grade
        predicted_grade = model.predict(new_student_df)[0]

        # Generate learning path and feedback
        feedback = generate_learning_path_and_feedback(new_student, predicted_grade)

        # Save the performance to the database
        student_id = 1  # Set the student ID appropriately
        save_performance(student_id, predicted_grade, 0)  # Assuming actual_grade is not known yet

        # Provide SHAP explanation
        shap_explanation = explain_prediction(model, new_student_df)

        return render_template('index.html', predicted_grade=predicted_grade, 
                               learning_path=feedback['learning_path'], 
                               feedback=feedback['feedback'],
                               shap_explanation=shap_explanation)
    
    return render_template('index.html')

# Route to upload CSV for batch prediction
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Load the pre-trained model and make predictions for the uploaded dataset
            model = load_model()
            predictions = model.predict(data)
            
            return render_template('batch_results.html', predictions=predictions)
    
    return render_template('upload.html')

if __name__ == '__main__':
    create_database()  # Ensure database is created
    app.run(debug=True)
