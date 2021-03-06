import cv2
import math
import json
import xlrd
import uuid
import numpy
import os
import glob
import random	
import pandas as pd
import numpy as np
import scipy.linalg
from tqdm import tqdm, trange
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
#from skimage.io import imread, imsave
from keras.models import Model
import xml.etree.ElementTree as ET  
from keras.utils import plot_model
import numpy.linalg as linalg
from keras.optimizers import Adam
from osgeo import gdal
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from classification_models.resnet import ResNet18, preprocess_input
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, ZeroPadding2D, Dense, Dropout, Activation


_EPS = numpy.finfo(float).eps * 4.0

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.
    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0
    """
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)

def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True
    """
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    direction = numpy.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    point = numpy.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)

def decompose_matrix(matrix):
	"""Return sequence of transformations from transformation matrix.

	matrix : array_like
	Non-degenerative homogeneous transformation matrix

	Return tuple of:
	scale : vector of 3 scaling factors
	shear : list of shear factors for x-y, x-z, y-z axes
	angles : list of Euler angles about static x, y, z axes
	translate : translation vector along x, y, z axes
	perspective : perspective partition of matrix

	Raise ValueError if matrix is of wrong type or degenerative.

	>>> T0 = translation_matrix([1, 2, 3])
	>>> scale, shear, angles, trans, persp = decompose_matrix(T0)
	>>> T1 = translation_matrix(trans)
	>>> numpy.allclose(T0, T1)
	True
	>>> S = scale_matrix(0.123)
	>>> scale, shear, angles, trans, persp = decompose_matrix(S)
	>>> scale[0]
	0.123
	>>> R0 = euler_matrix(1, 2, 3)
	>>> scale, shear, angles, trans, persp = decompose_matrix(R0)
	>>> R1 = euler_matrix(*angles)
	>>> numpy.allclose(R0, R1)
	True

	"""
	M = numpy.array(matrix, dtype=numpy.float64, copy=True).T
	if abs(M[3, 3]) < _EPS:
		raise ValueError('M[3, 3] is zero')
	M /= M[3, 3]
	P = M.copy()
	P[:, 3] = 0.0, 0.0, 0.0, 1.0
	if not numpy.linalg.det(P):
		raise ValueError('matrix is singular')
	scale = numpy.zeros((3, ))	
	shear = [0.0, 0.0, 0.0]
	angles = [0.0, 0.0, 0.0]
	if any(abs(M[:3, 3]) > _EPS):
		perspective = numpy.dot(M[:, 3], numpy.linalg.inv(P.T))
		M[:, 3] = 0.0, 0.0, 0.0, 1.0
	else:
		perspective = numpy.array([0.0, 0.0, 0.0, 1.0])
	translate = M[3, :3].copy()
	M[3, :3] = 0.0
	row = M[:3, :3].copy()
	scale[0] = vector_norm(row[0])
	row[0] /= scale[0]
	shear[0] = numpy.dot(row[0], row[1])
	row[1] -= row[0] * shear[0]
	scale[1] = vector_norm(row[1])
	row[1] /= scale[1]
	shear[0] /= scale[1]
	shear[1] = numpy.dot(row[0], row[2])
	row[2] -= row[0] * shear[1]
	shear[2] = numpy.dot(row[1], row[2])
	row[2] -= row[1] * shear[2]
	scale[2] = vector_norm(row[2])
	row[2] /= scale[2]
	shear[1:] /= scale[2]
	if numpy.dot(row[0], numpy.cross(row[1], row[2])) < 0:
		numpy.negative(scale, scale)
		numpy.negative(row, row)
	angles[1] = math.asin(-row[0, 2])
	if math.cos(angles[1]):
		angles[0] = math.atan2(row[1, 2], row[2, 2])
		angles[2] = math.atan2(row[0, 1], row[0, 0])
	else:
		# angles[0] = math.atan2(row[1, 0], row[1, 1])
		angles[0] = math.atan2(-row[2, 1], row[1, 1])
		angles[2] = 0.0
	return scale, shear, angles, translate, perspective


class AirBackBone:
	def __init__(self, img_W = 2048, img_H = 2048, dataDirectory = None):
		self.dataDirectory = dataDirectory
		self.img_W = img_W
		self.img_H = img_H
		self.resnet_M = None
		self.classes = 1000
		self.model = None
		self.flightName = None
		self.flightCode = None
		self.xmlPath = '/root/sharedfolder/My Files/Public/ashutosh/ウマ追加データ/ウマ追加データ/xml/'
		self.horLocDir = '/root/sharedfolder/My Files/Public/ashutosh/Horse/ProcessedData/locationData/'
		self.Datapath = '/root/sharedfolder/My Files/Public/ashutosh/'

	def CreateResnet(self):
		self.resnet_M = ResNet18((512, 512, 3), weights='imagenet', classes = 1000)
		
	def step_decay(self, epoch):
		if epoch <= 5: 
			return 1e-4
		elif epoch > 5 and epoch <= 10: 
			return 1e-5
		elif epoch > 10 and epoch <=100: 
			return 1e-6
		else: 
			return 1e-7

	def createModel(self):
		self.CreateResnet()
		self.resnet_M.summary()
		for i in range(90):
			if i >= 68:
				self.resnet_M.layers.pop()
			else:
				self.resnet_M.layers[i].trainable = False
		resLast = self.resnet_M.layers[66]
		# CNN = Conv2D(512, kernel_size=(3, 3), padcv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)ding='same', activation='relu')(self.resnet_M.layers[-1].output) 
		MLP = Conv2D(1024, kernel_size=(1, 1), padding='same', activation='relu')(
			self.resnet_M.layers[-1].output
		) 
		DROP = Dropout(0.5, noise_shape=None, seed=None)(MLP)
		MLP1 = Conv2D(2, kernel_size=(1, 1), padding='same')(DROP)
		SMAX = Activation('softmax')(MLP1)
		self.model = Model(inputs = self.resnet_M.input, outputs = SMAX)
		opt = Adam(
			lr=0.0001, decay=0.5, amsgrad=False
		)	#weight decay 0.0001
		self.model.compile(
			optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy']
		)
		self.model.summary()


	def generate_data(self, directory, batch_size):
		"""Replaces Keras' native ImageDataGenerator."""
		i = 0
		file_list = os.listdir(directory + 'Input')
		while True:
			image_batch = []
			image_mask = []
			while(len(image_mask)!=batch_size):
				if i == len(file_list):
					i = 0
					random.shuffle(file_list)
				sample = file_list[i]
				i += 1
				x = random.randint(1,2048)
				y = random.randint(1,2048)
				try:
					image = self.partialRanCrop(cv2.imread(directory + 'Input/' + sample), [x, y], 256, 2048, 2048)
					border = self.partialRanCrop(cv2.imread(directory + 'border_Mask/' + sample[:-3]+sample[-4:], cv2.IMREAD_GRAYSCALE), [x//64, y//64], 16, 128, 128)
					center = self.partialRanCrop(cv2.imread(directory + 'center_Mask/' + sample[:-3]+sample[-4:], cv2.IMREAD_GRAYSCALE), [x//64, y//64], 16, 128, 128)
				except:
					continue
				image_batch.append(image)
				image_mask.append([np.array(border), np.array(center)])				

			data_gen_args = dict(featurewise_center=True,
					     featurewise_std_normalization=True,
					     rotation_range=90,
					     width_shift_range=0.1,
					     height_shift_range=0.1,
					     zoom_range=0.2)
			image_datagen = ImageDataGenerator(**data_gen_args)
			mask_datagen = ImageDataGenerator(**data_gen_args)

			# Provide the same seed and keyword arguments to the fit and flow methods
			seed = 1
			'''image_datagen.fit(image_batch, augment=True, seed=seed)
			mask_datagen.fit(image_mask[0], augment=True, seed=seed)
			mask_datagen.fit(image_mask[1], augment=True, seed=seed)

			image_generator = image_datagen.flow_from_directory(
			    '/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/Input/',
			    class_mode=None,
			    seed=seed)

			border_mask_generator = mask_datagen.flow_from_directory(
			    '/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/border_Mask/',
			    class_mode=None,
			    seed=seed)

			center_mask_generator = mask_datagen.flow_from_directory(
			    '/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/center_Mask/',
			    class_mode=None,
			    seed=seed)

			# combine generators into one which yields image and masks
			train_generator = zip(image_generator, [border_mask_generator, center_mask_generator])
			print(train_generator.shape)'''
			yield np.array(image_batch), np.reshape(image_mask, (batch_size, 32, 32 ,2))


	def fit(self):
		lrate = LearningRateScheduler(self.step_decay)
		self.createModel()
		#In, Gt = self.loadData()
		batch_size = 32
		#x_train, x_test, y_train, y_test = train_test_split(In, Gt, test_size=0.33)
		checkpoint = ModelCheckpoint('checkpoint.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		self.model.fit_generator(self.generate_data('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/', batch_size), epochs=5, steps_per_epoch=len(os.listdir('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/Input/')) // batch_size, callbacks = [lrate, checkpoint])

		# here's a more "manual" example
		'''for e in range(epochs):
			print('Epoch', e)
			batches = 0
			for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
				model.fit(x_batch, y_batch, callbacks = [lrate])
				batches += 1
				if batches >= len(x_train) / 32:
					# we need to break the loop by hand because
					# the generator loops indefinitely
					break'''
		self.model.save_weights('detector.hdf5')

	def predict(self):
		self.createModel()
		self.model.load_weights('detector.hdf5')
		count = -1
		for filename in glob.glob('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/marked/*.jpg'):
			x = random.randint(1,2048)
			y = random.randint(1,2048)
			image = self.partialRanCrop(cv2.imread('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/Input/'+ filename[-40:]), [x, y], 256, 2048, 2048)
			count += 1
			pre = self.model.predict(np.reshape(image, (1, 512, 512, 3)))		
			cv2.imwrite(filename[-40:], np.reshape(pre, (2, 32 ,32))[1])
			if count == 10:
				break;		

	def partialRanCrop(self, img, coordinates, windowLen, origW, origH):
		y = coordinates[1]
		x = coordinates[0]
		if origW - x < windowLen:
			x_R = origW - 1
			x_L = x - (windowLen + (windowLen - (origW - 1 - x)))
		elif x < windowLen:
			x_L = 0
			x_R = 2*windowLen
		else: 
			x_R = x + windowLen
			x_L = x - windowLen
		if origH - y < windowLen:
			y_D = origH - 1
			y_U = y - (windowLen + (windowLen - (origH - 1 - y)))
		elif y < windowLen:
			y_U = 0
			y_D = 2*windowLen
		else: 
			y_D = y + windowLen
			y_U = y - windowLen
		newX = x - x_L
		newY = y - y_U
		return img[y_U: y_D, x_L: x_R]
		
	def processDataFile(self):
		with open('/home/ghost/savmap_dataset_v2/savmap_annotations_2014.geojson') as f:
			anno = json.load(f)
		gt = []
		sameImg = []
		count = 0

		for i in range(len(anno['features'])):
			if (
				anno['features'][i]['properties']['IMAGEUUID'] == anno['features'][i+1]['properties']['IMAGEUUID']
			):
				sameImgTag.append(anno['features'][i]['properties']['TAGUUID'])
				imageName = anno['features'][i]['properties']['IMAGEUUID']
				continue
			print('[{}] '.format(count)+imageName+' .... ', end = '')
			img = imread('/home/ghost/savmap_dataset_v2/'+ imageName + '.JPG')
			coordinates = np.array(anno['features'][i]['geometry']['coordinates'][0])
			x_centre = int(np.average(coordinates, axis = 0)[0])
			y_centre = int(np.average(coordinates, axis = 0)[1])
			ROI , newCoord = self.partialRanCrop(
				np.array(img), [y_centre, x_centre], 256, 4000, 3000
			)
			#ROI = cv2.circle(ROI,(newCoord[0], newCoord[1]), 25, (0,255,0), 3)
			dummy , gtCoord = self.partialRanCrop(
				ROI, [newCoord[0], newCoord[1]], 16, 512, 512
			)
			gt = np.empty((32, 32), dtype=np.float64)
			print(gtCoord)
			gt[gtCoord[0]-8: gtCoord[0]+8, gtCoord[1]-8:gtCoord[1]+8] = 1 
			'''gt[gtCoord[0]-1, gtCoord[1]-1] = 1	
			gt[gtCoord[0]-1, gtCoord[1]+1] = 1	
			gt[gtCoord[0]+1, gtCoord[1]-1] = 1	
			gt[gtCoord[0]+1, gtCoord[1]+1] = 1'''	
			#for i in range(len(coordinates)):
			#	img = cv2.circle(img,(int(coordinates[i][0]), int(coordinates[i][1])), 5, (0,255,0), 3)
			#cv2.imwrite("/media/ghost/DATA/Material/AirHorse Identification/Airbus/test/" + tag + ".jpg", img)
			cv2.imwrite(
				"/media/ghost/DATA/Material/AirHorse Identification/Airbus/gt/"
				 + tag
				 + ".jpg",
				gt
			)
			print("done")
			count = count + 1
			sameImg = []
			'''cv2.imwrite("/media/ghost/DATA/Material/AirHorse Identification/Airbus/processedData/" + tag + ".jpg", ROI)
			gt.append([tag, newCoord])


		with open('gt.json', 'w') as outfile:
			json.dump(gt, outfile)
		print(len(set(loc)))'''

	def dataProcessor(self):	
		def isRotationMatrix(R) :
			Rt = np.transpose(R)
			shouldBeIdentity = np.dot(Rt, R)
			print(Rt)
			I = np.identity(3, dtype = R.dtype)
			n = np.linalg.norm(I - shouldBeIdentity)
			return n < 1e-6

		def prepareGtTags():	
			df = pd.read_excel(
				self.Datapath + 'Horse_photo/position_horse_WGS84.xlsx'
			)
			df = np.array(df.astype(str).groupby('code').agg(','.join).reset_index())
			print("Preparing Horse Positions")
			for flight in trange(len(df)):
				#print('[{}] '.format(hor)+'........................................', end = '')
				#print(str(int(horSheet.row_values(hor)[1])))
				horsX = [(float(hor)) for hor in df[flight][3].split(',')]
				horsY = [(float(hor)) for hor in df[flight][4].split(',')]
				tag = [(str(hor)) for hor in df[flight][1].split(',')]
				#print('done')
				np.savez_compressed(
					self.horLocDir + df[flight][0],
					gt_Tag = tag,
					horPos = [horsX, horsY]
				)

		def listTif():
			tifs = []
			newTif = []
			for root, dirs, files in os.walk('/root/sharedfolder/My Files/Public/ashutosh/' + 'Horse_photo/'):
				for file in files:
					if file.endswith(".tif"):
						tifs.append(root + '/' + file)
			for tif in tifs:
				if tif.split('/')[-1][:-4][0] == '2':
					newTif.append(tif)

			with open(
				self.Datapath + 'Horse/ProcessedData/TifData/tif.json', 'w'
			) as outfile:
				json.dump(newTif, outfile)

		def flight2Code():
			if self.flightName[17] == '_':
				self.flightCode = self.flightName[6: 9] + '0' + self.flightName[16]
			else:
				self.flightCode = self.flightName[6: 9] + self.flightName[16:18]

		#assert(isRotationMatrix(np.array(R)[0:3,0:3]))
		listTif()

		#with open(
		#	self.Datapath + 'Horse/ProcessedData/exception.json', 'w'
		#) as outfile:
		#	json.dump([], outfile)
		count = 0
		prepareGtTags()
		with open(
			self.Datapath + 'Horse/ProcessedData/TifData/tif.json'
		) as f:
			tifs = json.load(f)
		count = 0
		#print('***************| Processing TIF files |*********************')
		for tif in tqdm(tifs):
			with open(
				self.Datapath + 'Horse/ProcessedData/TifData/tif.json', 'w'
			) as outfile:
				json.dump(tifs, outfile)
			with open(
				self.Datapath + 'Horse/ProcessedData/state.json', 'w'
			) as outfile:
				json.dump(tifs, outfile)
			self.flightName = tif.split('/')[-1][:-4]
			#print('[{}] '.format(count)+' ........................................... ', end = '')
			driver = gdal.GetDriverByName('GTiff')
			dataset = gdal.Open(tif)
			band = dataset.GetRasterBand(1)
			cols = dataset.RasterXSize
			rows = dataset.RasterYSize
			transform = dataset.GetGeoTransform()
			xOrigin = transform[0]
			yOrigin = transform[3]
			pixelWidth = transform[1]
			pixelHeight = -transform[5]
			data = band.ReadAsArray(0, 0, cols, rows)
			try:
				img  = imread(tif, plugin='tifffile')
				pass
			except:
				print('Exception too large tif')
				with open(
					self.Datapath + 'Horse/ProcessedData/exception.json'
				) as f:
					exception = json.load(f)
				tifs.remove(tif)
				exception.append(self.flightName)
				with open(
					self.Datapath + 'Horse/ProcessedData/exception.json', 'w'
				) as outfile:
					json.dump(exception, outfile)
				continue

			flight2Code()
			try:
				horses = np.load(self.horLocDir + self.flightCode +'.npz') 
			except:
				print('Exception too large tif')
				with open(
					self.Datapath + 'Horse/ProcessedData/exception.json'
				) as f:
					exception = json.load(f)
					tifs.remove(tif)
				exception.append(self.flightName)
				with open(
					self.Datapath + 'Horse/ProcessedData/exception.json', 'w'
				) as outfile:
					json.dump(exception, outfile)
				continue
			horseFlag = np.zeros((len(horses['gt_Tag']), 1), dtype='bool')

			if self.flightName[-3] == '_':
				self.flightName = self.flightName[0:27] + self.flightName[28:]	

			try:	
				tree = ET.parse(self.xmlPath + self.flightName + '.xml') 
				pass
			except: 
				print('Exception horse location not found')
				with open(
					self.Datapath + 'Horse/ProcessedData/exception.json'
				) as f:
					exception = json.load(f)
				exception.append(self.flightName)
				tifs.remove(tif)
				with open(
					self.Datapath + 'Horse/ProcessedData/exception.json', 'w'
				) as outfile:
					json.dump(exception, outfile)
				continue

			root = tree.getroot()

			#print('                     ******************| Processing Center in the TIF file |*******************')
			for center in trange(0, len(root[0][1])):
				#print('[{}] '.format(center)+'                      ........................................... ', end = '')
				multiHorse = []
				multiTag = []
				imagId = uuid.uuid4()
				try:
					if len(root[0][1][center][1].attrib) == 4:
						coordIndex = 1
					else:
						coordIndex = 2
					pass
				except:
					continue
				try:
					imagCenGpsX = float(root[0][1][center][coordIndex].attrib['x'])
					imagCenGpsY = float(root[0][1][center][coordIndex].attrib['y'])
					pass
				except:
					print('Exception horse location not found')
					with open(
						self.Datapath + 'Horse/ProcessedData/exception.json'
					) as f:
						exception = json.load(f)
					exception.append(self.flightName)
					with open(
						self.Datapath + 'Horse/ProcessedData/exception.json', 'w'
					) as outfile:
						json.dump(exception, outfile)
					continue
			
				imagCenPixX = int((imagCenGpsX - xOrigin) / pixelWidth) + random.randint(1,1000)
				imagCenPixY = int((yOrigin - imagCenGpsY) / pixelHeight) + random.randint(1,1000)
				#Transformation = (root[0][1][center][0].text)
				#Transformation = [(float(matrix)) for matrix in Transformation.split()]
				#Transformation_Inv = np.linalg.inv(np.reshape(np.array(Transformation), (4,4)))
				#scale, shear, angle, translate, pros = (decompose_matrix(Transformation_Inv))
				#rotation_matrix = cv2.getRotationMatrix2D((imagCenPixX, imagCenPixY), math.degrees(angle[0]), 1)
				#rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[0], img.shape[1]))
				#imagCenPixX = int(imagCenPixX - translate[0])
				if (imagCenPixX + self.img_W//2 > img.shape[1] 
					or imagCenPixX - self.img_W//2 < 0
					or imagCenPixY - self.img_H//2 < 0
					or imagCenPixY + self.img_H//2 > img.shape[0]
				):
					continue
				crop = img[
					imagCenPixY - self.img_H//2: imagCenPixY + self.img_H//2,
					imagCenPixX - self.img_W//2: imagCenPixX + self.img_W//2,
					0:3
				]
				brown_lo=np.array([250,250,250])
				brown_hi=np.array([255, 255, 255])
				mask=cv2.inRange(crop,brown_lo, brown_hi)
				crop[mask>0]=crop[1024,1024, :]
				marked = crop

				#print('********************| Processing Horses with respect to Center |*******************')
				for horse in trange(0, len(horses['gt_Tag'])):
					#print('[{}] '.format(horse)+' ........................................... ', end = '')
					horseGpsX = float(horses['horPos'][0][horse])
					horseGpsY = float(horses['horPos'][1][horse])
					horsePixX = int((horseGpsX - xOrigin) / pixelWidth)
					horsePixY = int((yOrigin - horseGpsY) / pixelHeight)
					horseImgX = horsePixX - (imagCenPixX - self.img_W//2)
					horseImgY = horsePixY - (imagCenPixY - self.img_H//2)
					if (
						horseImgX < self.img_W
						and horseImgY < self.img_H
						and horseImgX > 0
						and horseImgY > 0
					):
						marked = cv2.circle(marked, (horseImgX, horseImgY), 25, (0,255,0), 3)
						horseFlag[horse] = 1
						multiHorse.append([horseImgX, horseImgY])
						multiTag.append(horses['gt_Tag'][horse])
				if(len(multiHorse)==0):			
					imsave(
						self.Datapath
						+ "Horse/ProcessedData/newData/backGroundData/"
						+ str(imagId)
						+ ".jpg",
						crop
					)
					'''cv2.imwrite(
						self.Datapath
						+ "Horse/ProcessedData/newData/marked/"
						+ str(imagId)
						+ ".jpg",
						marked
					)
					np.savez_compressed(
						self.Datapath
						+ 'Horse/ProcessedData/newData/gt_Pos/'
						+ str(imagId),
						gt_Tag = multiTag,
						horseCoordinates = multiHorse,
					)'''
		
					


					
			tifs.remove(tif)
					#print('done')
				#print('done')
				#count = count + 1
				#print('done')
	def generateGt(self):
		#count = 0

		for file in tqdm(glob.glob('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/Input/*.jpg')):
			im = cv2.imread(file)
			try:
				l = np.load('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/gt_Pos/' + file[-40:-3]+'npz')
			except:
				continue
			center = np.full((128, 128), 0)
			border = np.zeros((128, 128))
			for i in range(len(l['horseCoordinates'])):
				center[int(l['horseCoordinates'][i][1]/16)][int(l['horseCoordinates'][i][0]/16)] = 255
				border[int(l['horseCoordinates'][i][1]/16)-8:int(l['horseCoordinates'][i][1]/16)+8, int(l['horseCoordinates'][i][0]/16) - 8: int(l['horseCoordinates'][i][0]/16)+8] = 255
				cv2.imwrite('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/border_Mask/'+ file[-40:-3] + '.jpg', border)
				cv2.imwrite('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/center_Mask/'+ file[-40:-3] + '.jpg', center)


						


	def loadData(self):
		image_list = []
		anno = []
		for filename in tqdm(glob.glob( '/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/Input/*.jpg')):
			gt = []
			im = cv2.imread(filename)
			image_list.append(im)
			try:
				gt.append(cv2.imread('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/gt_Masks/'+filename+'border.jpg'))
				gt.append(cv2.imread('/run/user/1000/gvfs/sftp:host=163.221.84.106,user=ashutosh/mnt/My Files/Public/ashutosh/Horse/ProcessedData/newData/gt_Masks/'+filename+'center.jpg'))
				anno.append(gt)
			except:
				continue

		return image_list, anno
		

		
a = AirBackBone()
#a.createModel()
a.predict()
#a.resnet_M.summary()
