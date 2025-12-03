Phishing URL Detector - Simple Instructions

This project is a small machine learning program that checks if a URL is safe or phishing. 
The model is already trained, so you do NOT need to train anything unless you want to. 
You just need to update the file paths and run the program.

---------------------------------------------------------
WHAT IS IN THE PROJECT FOLDER
---------------------------------------------------------

You should have these files:

app.py
model.pkl
vectorizer.pkl
model_train.py   (you only need this if you want to retrain the model)
templates/
    index.html

app.py runs the website
model.pkl is the trained AI model
vectorizer.pkl is used to turn URLs into numbers
model_train.py is only for training again
index.html is the webpage

---------------------------------------------------------
STEP 1: INSTALL THE REQUIRED PYTHON LIBRARIES
---------------------------------------------------------

Open a terminal inside this project folder and run this:

pip install flask scikit-learn pandas numpy scipy

This installs everything the program needs.

---------------------------------------------------------
STEP 2: FIX THE FILE PATHS IN app.py
---------------------------------------------------------

You MUST update two lines so the program knows where your model files are.

In app.py, change these two lines:

model = pickle.load(open("C:/Users/confl/OneDrive/Desktop/phishing-ai-project/model.pkl", "rb"))
vectorizer = pickle.load(open("C:/Users/confl/OneDrive/Desktop/phishing-ai-project/vectorizer.pkl", "rb"))

Change the paths so they match where YOUR project folder is located.

Example:

model = pickle.load(open("C:/Users/YourName/Desktop/PhishingProject/model.pkl", "rb"))
vectorizer = pickle.load(open("C:/Users/YourName/Desktop/PhishingProject/vectorizer.pkl", "rb"))

Once the paths are correct, the app will work.

---------------------------------------------------------
STEP 3: RUN THE PROGRAM
---------------------------------------------------------

Now run this:

python app.py

After it starts, it will show something like:

Running on http://127.0.0.1:5000/

Open that link in your browser.

---------------------------------------------------------
STEP 4: USE THE WEBPAGE
---------------------------------------------------------

Type any URL in the text box.
Click the "Check URL" button.

The page will tell you if the URL is:

Phishing (Unsafe)
or
Legitimate (Safe)

The AI is already trained, so there is nothing else you need to do.

---------------------------------------------------------
OPTIONAL: HOW TO RETRAIN THE MODEL (ONLY IF YOU WANT TO)
---------------------------------------------------------

You do NOT need to do this unless you want to use a different dataset.

If you DO want to retrain, follow these steps:

1. Get a CSV file that has two columns:
   url
   label (0 = safe, 1 = phishing)

2. Open model_train.py and go to this line:

   csv_path = "C:/Users/confl/OneDrive/Desktop/phishing-ai-project/data/phishing_URL.csv"

   Change it to your CSV file path.

3. Run:

   python model_train.py

This will make NEW model.pkl and vectorizer.pkl files.

4. Update the paths in app.py again so it uses the new files.

---------------------------------------------------------
SUMMARY
---------------------------------------------------------

1. Install the libraries
2. Update the file paths in app.py
3. Run app.py
4. Use the webpage to test URLs

You do NOT need to train anything unless you want to.

Now You Should Be All Set
