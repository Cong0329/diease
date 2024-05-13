from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Lấy giá trị từ form
        disease = request.form['Disease']
        fever = request.form['Fever']
        cough = request.form['Cough']
        fatigue = request.form['Fatigue']
        difficulty_breathing = request.form['DifficultyBreathing']
        age = request.form['Age']
        gender = request.form['Gender']
        blood_pressure = request.form['BloodPressure']
        cholesterol_level = request.form['CholesterolLevel']
        
        # In ra các giá trị
        print("Disease:", disease)
        print("Fever:", fever)
        print("Cough:", cough)
        print("Fatigue:", fatigue)
        print("Difficulty Breathing:", difficulty_breathing)
        print("Age:", age)
        print("Gender:", gender)
        print("Blood Pressure:", blood_pressure)
        print("Cholesterol Level:", cholesterol_level)
        
        return "Form submitted successfully!"
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)

