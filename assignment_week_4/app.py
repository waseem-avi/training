from flask import Flask

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return 'Hello, World! This is the homepage.'

# Define a route for a custom page
@app.route('/about')
def about():
    return 'This is the about page.'

# Run the application
if __name__ == '__main__':
    app.run(debug=True)