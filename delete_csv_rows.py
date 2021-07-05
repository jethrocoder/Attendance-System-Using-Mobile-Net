import csv
f = open('student_details.csv', "r+")
lines = f.readlines()
lines.pop()
f = open('student_details.csv', "w+")
f.writelines(lines)