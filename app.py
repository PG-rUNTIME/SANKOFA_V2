# To run this file:
# 1. Install Flask: pip install Flask
# 2. Save this file as 'app.py' and the HTML file as 'templates/index.html'
# 3. Run from terminal: python app.py

from flask import Flask, render_template

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    """
    Renders the main chat application HTML file.
    Since the HTML contains all necessary CSS and JavaScript, 
    we only need to render the template.
    """
    # Flask will look for 'index.html' inside the 'templates' folder.
    return render_template('index.html')

if __name__ == '__main__':
    # Running on port 5000 by default, which is convenient when
    # your FastAPI backend is running on port 8000.
    print("Flask server running at http://127.0.0.1:5000/")
    app.run(debug=True)