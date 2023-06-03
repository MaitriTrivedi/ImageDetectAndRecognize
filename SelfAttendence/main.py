import face_recognition
import os
from pathlib import Path
from datetime import date
import time

# Give path to known images folder
known_images_files_path = [os.path.abspath("known_faces/"+x) for x in os.listdir("known_faces/")]

# To create excel file for daily report with name as date
current_date = str(date.today())+".xlsx"

# Give path of the folder where you want to store the daily attendence report 
attendence_report_files_path = [os.path.abspath("AttendenceReport/"+x) for x in os.listdir("AttendenceReport/")]

# If excel sheet of particular day is not there then it will be generated 
# Else if it is already there then it will be used to store attendence of that particular day
if not any(current_date in file_path for file_path in attendence_report_files_path):
    file_name = str(current_date)
    f = open("AttendenceReport/"+file_name, "x")
    name = "NAME"
    current_time = "TIME"
    d1 = "DATE"
    f.writelines(f'{name},{current_time},{d1}')

# In unknown_image_file you need to add image of the whose attendence you want to be added
# It can be added through live video or any other way you want 
# For that you need to change the input for unknown_image_file
unknown_image_file = "unknown_faces/einstien.jpeg"

for i in known_images_files_path :
    known_image_file =  i
    known_image = face_recognition.load_image_file(known_image_file)
    unknown_image = face_recognition.load_image_file(unknown_image_file)
    temp = Path(known_image_file).stem
    # print(temp)

    elonMusk_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    result = face_recognition.compare_faces([elonMusk_encoding], unknown_encoding)
    if result[0]==True :
        with open('AttendenceReport/'+current_date,'r+') as f:
            name = temp
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            print(nameList)
            if name not in nameList:
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)
                print(current_time)
                today = date.today()
                d1 = today.strftime("%d/%m/%Y")
                f.writelines(f'\n{name},{current_time},{d1}')

        print(result)
        # break

    
