import pandas as pd
import os
import numpy as np
# import face_data_collect as f
# import face_recognition as fr
from csv import writer
from pca import reduce_dimentions

source_file = './val_data/'
destination_file = './pca_val_data/'


for fx in os.listdir(source_file):
	if fx.endswith('.npy'):
		pca_dataset = []
		raw_data = np.load(source_file+fx)
		print('raw shape ',raw_data.shape,end='\t')
		for img in raw_data:
			img = img.reshape((100,100))
			
			red = reduce_dimentions(img)
			pca_dataset.append(red)
			
		# Convert our face list array into a numpy array
		pca_dataset = np.asarray(pca_dataset)
		pca_dataset = pca_dataset.reshape((pca_dataset.shape[0],-1))
		print('pca shape ',pca_dataset.shape)
		np.save(destination_file+fx , pca_dataset)

