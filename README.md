Step 1: Clone the Project from GitHub
1.	Install Git if you haven't already (from git-scm.com).
2.	Open your terminal or command prompt and clone your repository:
git clone https://github.com/Dylan-Hoang03/DocumentMind.git
cd your-repo

Alternative: You can just download the zip file from github
________________________________________
Step 2: Install Python (if not already installed)
1.	Download Python from python.org. Ensure you check the box to add Python to the system PATH during installation.
2.	Verify the installation:
python --version
This should return the version of Python that is installed.
Should be version 3.11 atleast
________________________________________
Step 3: Create and Activate a Virtual Environment
A virtual environment isolates your dependencies from the system-wide Python installation, which helps avoid version conflicts.
1.	Create a virtual environment inside your project folder:
python -m venv venv
2.	Activate the virtual environment:
venv\Scripts\activate
3.	After activation, your terminal should show (venv) at the beginning of the prompt.
________________________________________
Step 4: Install the Required Python Packages
1.	With your virtual environment activated, install the required dependencies from the requirements.txt file:
pip install -r requirements.txt
This will install all the Python dependencies needed for the backend to work.
________________________________________
Step 5: Set Up Ollama and LM Studio
You need Ollama and LM Studio to run the LLM-related tasks in your backend.
1. Install Ollama:
1.	Visit the Ollama website and download the Ollama client for your operating system.
2.	Follow the installation instructions to get Ollama up and running.
3.	Once installed, you can test it by running:
ollama --version
2. Install LM Studio:
1.	Visit the LM Studio website and download the appropriate version for your operating system.
2.	Install it following the provided instructions.
3.	After installation, you can test if LM Studio is running by launching it from your terminal or system menu and verifying it's accessible via http://localhost:1234 (or another port based on your configuration).
Make sure LM Studio is correctly configured and running locally.
________________________________________
Step 6: Run the Backend
1.	Run your Flask backend by executing the following command:
python app.py
This should start the Flask server, and you should see output like Running on http://127.0.0.1:5000/.
________________________________________
Step 7: Test the Project
1.	Open your frontend (if it's a React app):
npm start (if developing)
npm run build (if deploying)
This should open your app in the browser.
2.	Interacting with the Backend:
o	Use the frontend to test the file upload, querying, and search functionalities.
o	Ensure that your backend is processing PDFs, creating FAISS indexes, and responding to queries from the frontend.


Run server directions: 
cd server
venv\Scripts\activate   
python app.py

Run client directions: 
cd client
npm start