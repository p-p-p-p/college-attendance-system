import os
revoked_list = ["database.csv","database.sqlite","name.sqlite","attendance.sqlite","attendance_report.csv","attendance_report.xlsx","database.csv"]
for i in revoked_list:
    try:
        os.remove(i)
    except:
        pass
