

![](https://github.com/p-p-p-p/college-attendance-system/blob/main/banner.png)

# College Attendance System
The "College Attendance System" is a powerful web application designed to effectively manage attendance records for both students and teachers in college courses. With this system, students and teachers can easily register for courses and the system will capture their name, ID, course name, and face embedding using a 30-second video from their webcam during registration.

The application offers several convenient features such as a tab to clean and prepare the dataset, another tab for building a support vector machine (SVM) model to accurately classify student/teacher faces, and a real-time attendance tab for taking attendance during classes.

The attendance records are stored in both sqlite and CSV formats, making it easy to view and analyze the data.With plans to integrate MongoDB in the future for efficient data storage. Additionally, the system provides an attendance analysis tab that allows users to review attendance data and generate reports.

By leveraging advanced technologies such as streamlit, single-shot detector (SSD)  and google facenet , the College Attendance System offers an accurate and efficient way to track attendance. With its automated attendance tracking capabilities, this system can help colleges comply with regulations, maintain accurate records, and improve overall efficiency.


This application is built using Python 3.10 .

## Run everything on google colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/p-p-p-p/college-attendance-system-/blob/main/college_attendance_system%20.ipynb)

## How to install 
####  Step 1: Git clone this repository
####  Step 2: cd ./college-attendance-system/
####  Step 3: Open terminal and copy paste this line
```
pip install -r requirements.txt
```
or
```
Run install.py
```
annd
```
rull download.py
```

####  Step 4: Run app.py
```
streamlit run app.py
```
### Run run.py

## Generate Unknown face class
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/p-p-p-p/college-attendance-system/blob/main/unknown_face.ipynb)

## Download unknow.csv file


## Homepage
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/homepage.png)
## Registration
![](https://github.com/p-p-p-p/college--attendance-system/blob/main/images/clip.gif)
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/registration.png)
## Realtime attendance for every subject
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/real.png)
## Sending confirmation mail in realtime
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/confirmation.jpg)
## Clean dataset and train model
![](https://github.com/p-p-p-p/college--attendance-system/blob/main/images/4.png)

## After the completion of attendance, analyze all of the attendance records.
![](https://github.com/p-p-p-p/college--attendance-system/blob/main/images/5.png)
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/ana.png)

## Send attendance records to teacher.
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/send.png)
## Received Excel file by mail
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/attend.jpg)
