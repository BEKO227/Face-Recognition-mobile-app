@echo off
rem Activate the Conda environment
call C:\Users\maraw\anaconda3\Scripts\activate.bat katna

rem Run the Python script
python "C:\Users\maraw\Desktop\Grad-Project\face_recognition_final\Final_Code.py"

rem Optional: Deactivate the Conda environment after execution
call conda deactivate
