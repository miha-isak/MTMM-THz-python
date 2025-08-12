

## Setup Instructions

1. Download all project files from GitHub.

2. Create a Python virtual environment:
   
   - **Windows (PowerShell):**  
     `python -m venv venv`  
     `.\venv\Scripts\Activate.ps1`
   
   - **Windows (cmd):**  
     `python -m venv venv`  
     `.\venv\Scripts\activate.bat`
   
   - **Linux / macOS:**  
     `python3 -m venv venv`  
     `source venv/bin/activate`
   
3. Install dependencies:  
   `pip install -r requirements.txt`

4. Run the project:  
   `python main.py`


## Optional Settings

For higher accuracy (with increased runtime) or faster execution (with potentially lower accuracy), you can adjust the following parameters:
PopSize – Population size. Increasing this improves accuracy but also increases runtime.
Maxit – Maximum number of generations. More iterations yield better results but take longer to compute.

You can also add your own data to the Data folder if needed. To do this:
1. Create a JSON file similar to the existing ones. You can add or remove layers by editing the thickness and refractive index values as needed.
2. Include two time-domain data files (as .txt files):
   - One for the reference measurement
   - One for the sample measurement
These files should contain the time-dependent experimental data and must also be placed in the Data folder.
