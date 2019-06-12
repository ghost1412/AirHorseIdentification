import cv2
import math
import json
import xlrd
import uuid
import numpy
import os
import numpy as np
import scipy.linalg
from skimage.io import imread, imsave
from keras.models import Model
import xml.etree.ElementTree as ET  
from keras.utils import plot_model
import numpy.linalg as linalg
from osgeo import gdal
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
#from classification_models.resnet import ResNet18, preprocess_input
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
		MLP1 = Conv2D(3, kernel_size=(1, 1), padding='same')(DROP)
		SMAX = Activation('Softmax')(CNN)
		self.model = Model(inputs = self.resnet_M.input, outputs = SMAX)
		opt = Adam(
			lr=0.0001, decay=0.5, amsgrad=False, momentum = 0.9
		)	#weight decay 0.0001
		self.model.compile(
			optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy']
		)
		self.model.summary()
		
	def fit(self):
		lrate = LearningRateScheduler(self.step_decay)
		In, Gt = self.loadData()
		x_train, x_test, y_train, y_test = train_test_split(In, Gt, test_size=0.33)
		datagen = ImageDataGenerator(
				featurewise_center=True,
				featurewise_std_normalization=True,
				rotation_range=20,
				width_shift_range=0.2,
				height_shift_range=0.2,
				horizontal_flip=True,
		)
		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		datagen.fit(x_train)

		# fits the model on batches with real-time data augmentation:
		model.fit_generator(
			datagen.flow(x_train, y_train, batch_size=32), 
			steps_per_epoch=len(x_train) / 32,
			epochs=epochs,
		)
		history = autoencoder.fit(
			x_train,
			x_train,
			epochs= 400,
			batch_size=30,
			shuffle=True,
			validation_data=(x_test, x_test),
			callbacks = [lrate],
		)

	def partialRanCrop(self, img, coordinates, windowLen, origW, origH):
		y = coordinates[0]
		x = coordinates[1]
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
		return img[y_U: y_D, x_L: x_R, :], [newX, newY]
		
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
			hors = xlrd.open_workbook(
				self.Datapath + 'Horse_photo/position_horse_WGS84.xlsx'
			)
			horSheet = hors.sheet_by_index(0) 
			for hor in range(horSheet.nrows):
				print('[{}] '.format(hor)+'........................................', end = '')
				horPos.append(
					[horSheet.row_values(hor)[3], horSheet.row_values(hor)[4]]
				)
				gt_Tag.append(horSheet.row_values(hor)[0])
				np.savez_compressed(
					self.horLocDir + str(horSheet.row_values(hor)[1]),
					gt_Tag = gt_Tag,
					horPos = horPos
				)
				print('done')

		def listTif():
			tifs = []
			for root, dirs, files in os.walk('/root/sharedfolder/My Files/Public/ashutosh/' + 'Horse_photo/'):
				for file in files:
					if file.endswith(".tif"):
						tifs.append(root + '/' + file)

			with open(
				self.Datapath + 'Horse/ProcessedData/TifData/tif.json', 'w'
			) as outfile:
				json.dump(tifs, outfile)

		def flight2Code():
			if self.flightName[17] == '_':
				self.flightCode = self.flightName[6: 8] + '0' + self.flightName[16]
			else:
				self.flightCode = self.flightName[6: 8] + self.flightName[16]

		horPos = []
		gt_Tag = []

		#assert(isRotationMatrix(np.array(R)[0:3,0:3]))
		listTif()
		with open(
			self.Datapath + 'Horse/ProcessedData/TifData/tif.json'
		) as f:
			tifs = json.load(f)

		count = 0

		prepareGtTags()
		horses = np.load(self.horLocDir + flightCode +'.npz')
		print('Processing TIF files')
		for tif in tifs:
			self.flightName = tif.split('/')[-1][:-4]
			if self.flightName[0] == '2':
				print('[{}] '.format(count)+' ............... ', end = '')
				print(self.flightName)
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
				img  = imread(tif)
				tree = ET.parse(self.xmlPath + self.flightName + '.xml')  
				flight2Code()
				root = tree.getroot()
				print('Processing Center in the TIF file')
				for center in range(len(root[0][1])):
					print('[{}] '.format(center)+' .... ', end = '')
					multiHorse = []
					imagId = uuid.uuid4()
					imagCenGpsX = float(root[0][1][center][2].attrib['x'])
					imagCenGpsY = float(root[0][1][center][2].attrib['y'])
					imagCenPixX = int((imagCenGpsX - xOrigin) / pixelWidth)
					imagCenPixY = int((yOrigin - imagCenGpsY) / pixelHeight)
					#Transformation = (root[0][1][center][0].text)
					#Transformation = [(float(matrix)) for matrix in Transformation.split()]
					#Transformation_Inv = np.linalg.inv(np.reshape(np.array(Transformation), (4,4)))
					#scale, shear, angle, translate, pros = (decompose_matrix(Transformation_Inv))
					#rotation_matrix = cv2.getRotationMatrix2D((imagCenPixX, imagCenPixY), math.degrees(angle[0]), 1)
					#print(cv2.getRotationMatrix2D((imagCenPixX, imagCenPixY), math.degrees(angle[0]), 1), cv2.getRotationMatrix2D((imagCenPixX, imagCenPixY), math.degrees(angle[1]), 1), cv2.getRotationMatrix2D((imagCenPixX, imagCenPixY), math.degrees(angle[2]), 1))
					#rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[0], img.shape[1]))
					#imagCenPixX = int(imagCenPixX - translate[0])
					#imagCenPixY = int(imagCenPixY - translate[1])
					crop = img[
						imagCenPixY - self.img_H//2: imagCenPixY + self.img_H//2,
						imagCenPixX - self.img_W//2: imagCenPixX + self.img_W//2
					]
					horseFlag = np.zeros((len(horses['gt_Tag']), 1), dtype='bool')

					print('Processing Horses with respect to Center')
					for horse in range(1, len(horses['gt_Tag'])):
						print('[{}] '.format(horse)+' .... ', end = '')
						horseGpsX = float(horses['horPos'][horse][0])
						horseGpsY = float(horses['horPos'][horse][1])
						horsePixX = int((horseGpsX - xOrigin) / pixelWidth)
						horsePixY = int((yOrigin - horseGpsY) / pixelHeight)
						horseImgX = horsePixX - (imagCenPixX - self.img_W//2)
						horseImgY = horsePixY - (imagCenPixY - self.img_H//2)
						if (
							horseImgX < self.imgY_W
							and horseImgY < self.img_H
							and horseImgX > 0
							and horseImgY > 0
						):
							cv2.circle(crop, (horseImgX, horseImgY), 25, (0,255,0), 3)
							horseFlag[horse] = 1
							multiHorse.append([horseImgX, horseImgY])
						#while(np.count_nonzero(horseFlag)):
						print('done')
					np.savez_compressed(
						self.Datapath
						+ 'Horse/ProcessedData/gt_Pos/'
						+ imagId,
						horseCoordinates = multiHorse,
					)
							
					cv2.imwrite(
						self.Datapath
						+ "Horse/ProcessedData/InputData/"
						+ imagId
						+ ".jpg",
						crop,
					)
					print('done')
				count = count + 1
				print('done')

	def loadData(self):
		for filename in glob.glob(
			'/media/ghost/DATA/Material/AirHorse Identification/Airbus/processedData/*.jpg'
		): 
    			im=Image.open(filename)
    			image_list.append(im)

		with open(
			'/media/ghost/DATA/Material/AirHorse Identification/Airbus/gt.json'
		) as f:
			anno = json.load(f)
		return image_list, anno
		

		
a = AirBackBone()
#a.createModel()
a.dataProcessor()
#a.resnet_M.summary()
