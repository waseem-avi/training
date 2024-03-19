from flask import Flask, render_template, request
from logic_csv import find_corrections

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return "Please upload both files."

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return "Please select both files."

        # Pass uploaded files to the function
        corrections = find_corrections(file1, file2)

        # You can then process the uploaded files as needed

        return render_template('index.html', corrections=corrections)

    return render_template('index.html')


# Run the application
if __name__ == '__main__':
    app.run(debug=True)