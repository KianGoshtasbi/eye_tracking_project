USE PYTHON VERSION BETWEEN 3.7-3.10

# EyeTracker

## Project Description
**EyeTracker** is a real-time eye landmark detection system built using **MediaPipe** and **OpenCV** in Python.  
It detects facial landmarks, extracts eye landmarks, calculates the **Eye Aspect Ratio (EAR)**, and visualizes the eyes with color-coded contours:  

- **Green:** Eyes open  
- **Red:** Eyes closed  


## Installation Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/eye_tracking_project.git
cd eye_tracking_project 
```
(Make sure you are in the project directory for the next steps)
2. **Set up a Python virtual environment:**
```bash
python -m venv venv
```
3. **Activate the virtual environment:**
For Windows(PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```
For Mac/linux
```bash
source venv/bin/activate
```
4. **Install required packages:**
```bash
pip install -r requirements.txt
```

Now you can run the eye tracker with
```bash
python eye_tracker.py
```
Press q to quit the program

Known Limitations are
1. Winking can cause program to not give expected results
2. Can run poorly on lower end machines (Was tested and built locally on Windows 10 with AMD Ryzen 7 5800x 8-core Processor, 16GB memory)
- If having trouble running localy can try configuring MediaPipe for GPU acceleration

Some usage examples that this program could be used for are
1. Counting the number of times someone has blinked
2. Seeing if someone is dozing off by checking how many frames their eyes have been closed for
3. Pure data analysis if someone wants to analyze average blinking rate of people (maybe to compare different demographics)
