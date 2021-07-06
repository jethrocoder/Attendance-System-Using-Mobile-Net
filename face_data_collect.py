import cv2
import pandas as pd
import numpy as np
from tkinter import messagebox
import tkinter as tk
import os
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
import pickle
from PIL import Image
import shutil

detector = MTCNN()
def register(txt,txt2):
	t = tk.Tk()
	t.geometry('+1050+120')
	t.configure(background='#122c57')
	l1 = tk.Label(t,text="taking 10 photos\n",fg='white',bg='#122c57')
	l1.pack()
	#Init Camera
	cap = cv2.VideoCapture(0)

	skip = 0
	face_data = []
	name = txt2.get().upper()
	roll_no = txt.get().upper()
	while True:
		ret,frame = cap.read()

		if ret==False:
			continue

		faces = detector.detect_faces(frame)
		if len(faces)==0:
			print('your face is not visible \n please get into the frame')
			continue
			
		x, y, w, h  = faces[0]['box']
		offset = 10
		cv2.rectangle(frame,(x-offset,y-offset),(x+w+offset,y+h+offset),(0,255,255),2)
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(254,254))
		face_section = cv2.cvtColor(face_section,cv2.COLOR_BGR2RGB)

		skip += 1
		face_data.append(face_section)
		l2 = tk.Label(t,text=str(len(face_data))+"\n",fg='white',bg='#122c57')
		l2.pack()				
		print(len(face_data))
	
		cv2.imshow("FRAME",frame)
		cv2.imshow("FACE",face_section)
		if skip%100 == 0:
			t.update()
		key_pressed = cv2.waitKey(1) & 0xFF
		if key_pressed == ord('q') or len(face_data) >= 200:
			t.destroy()
			break

	cap.release()
	cv2.destroyAllWindows()

	# Save this data into file system
	dataset_path='./train/'
	if os.path.isdir('{}{}'.format(dataset_path,roll_no)):
		shutil.rmtree('{}{}'.format(dataset_path,roll_no))
	os.mkdir('{}{}'.format(dataset_path,roll_no))
	print('New directory {} created at train folder'.format(roll_no))
	cnt=1
	for face in face_data:
		img=Image.fromarray(face)  
		path = '{}{}/{}.jpeg'.format(dataset_path,roll_no,cnt)
		cnt+=1
		img.save(path)
	print("Data Successfully save at "+dataset_path+roll_no)

	# Registering student in csv file
	with open('./saved/id_to_roll_f.pkl','rb') as f:
		try:
   	 		id_to_roll = pickle.load(f)
		except EOFError:
			id_to_roll = {}
	with open('./saved/roll_to_id_f.pkl','rb') as f:
		try:
   	 		roll_to_id = pickle.load(f)
		except EOFError:
			roll_to_id = {}

	model_id = len(id_to_roll)
	id_to_roll[model_id] = roll_no
	roll_to_id[roll_no] = model_id

	with open('./saved/id_to_roll_f.pkl','wb') as f:
   	 	pickle.dump(id_to_roll, f)
	with open('./saved/roll_to_id_f.pkl','wb') as f:
		pickle.dump(roll_to_id, f)
	
	row = np.array([roll_no,name,model_id]).reshape((1,3))
	df = pd.DataFrame(row) 
	# if file does not exist write header
	if not os.path.isfile('student_details.csv'):
	   df.to_csv('student_details.csv', header=['roll','name','model_id'],index=False)
	else: # else it exists so append without writing the header
	   df.to_csv('student_details.csv', mode='a', header=False,index=False)
		
	
	tk.messagebox.showinfo("Notification", "You have been registered successfully") 
