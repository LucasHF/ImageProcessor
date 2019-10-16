import cv2
import numpy as np
import math
from copy import deepcopy

def negativeFilter(image):
	result = deepcopy(image)
	for x in range(0,image.shape[0]):
		for y in range(0,image.shape[1]):
			result[x, y] = 255 - image[x,y]
	return result

def logFilter(image, constant):
	result = deepcopy(image)
	for x in range(0,image.shape[0]):
		for y in range(0,image.shape[1]):
			result[x, y] = constant * np.log1p(image[x, y])
	return result

def gammaFilter(image, constant, gamma):
	result = deepcopy(image)
	for x in range(0,image.shape[0]):
		for y in range(0,image.shape[1]):
			result[x, y] = constant * (image[x, y]**gamma)

	return result

#### NÃO ESTÁ FUNCIONANDO CORRETAMENTE
def convFilter(image, convMatrix, matrixSize):
	result = deepcopy(image)
	for x in range(0,image.shape[0]):
		for y in range(0,image.shape[1]):
			
			sumR = 0;
			sumG = 0
			sumB = 0

			for xConv in range(0, matrixSize):
				for yConv in range(0, matrixSize):
					xIndex = x - matrixSize//2 + xConv
					yIndex = y - matrixSize//2 + yConv

					if xIndex>=0 and xIndex<image.shape[0] and yIndex>=0 and yIndex<image.shape[1]:
						(b, g, r) = image[xIndex, yIndex]
						sumR += convMatrix[xConv][yConv] *  r
						sumG += convMatrix[xConv][yConv] *  g
						sumB += convMatrix[xConv][yConv] *  b
			if sumR>255:
				sumR = 255
			if sumG>255:
				sumG = 255
			if sumB>255:
				sumB = 255
			
			result[x, y] = (sumB, sumG, sumR)

	return result		
				

def mediaFilter(image, convMatrix, matrixSize):
	result = deepcopy(image)
	for x in range(0,image.shape[0]):
		for y in range(0,image.shape[1]):
			
			sumR = 0;
			sumG = 0
			sumB = 0

			for xConv in range(0, matrixSize):
				for yConv in range(0, matrixSize):
					xIndex = x - matrixSize//2 + xConv
					yIndex = y - matrixSize//2 + yConv

					if xIndex>=0 and xIndex<image.shape[0] and yIndex>=0 and yIndex<image.shape[1]:
						(b, g, r) = image[xIndex, yIndex]
						sumR += convMatrix[xConv][yConv] *  r
						sumG += convMatrix[xConv][yConv] *  g
						sumB += convMatrix[xConv][yConv] *  b

			divisor = matrixSize**2

			result[x, y] = (sumB//divisor, sumG//divisor, sumR//divisor)

	return result	

#Verificar Fórmula Correta!!!
def gaussianFormule(x, y, sigma): 
	return np.exp(-x*y/(2*sigma**2))

def gaussianFilter(image, matrixSize, sigma):
	result = deepcopy(image)
	mask = np.zeros((matrixSize, matrixSize))
	for x in range(0,matrixSize):
		for y in range(0, matrixSize):
			value = gaussianFormule(x-matrixSize//2, y-matrixSize//2, sigma)
			mask[x][y] = value

	return convFilter(image, mask, matrixSize)

		


############ HISTOGRAMA ################

def hist(image): 
	resultR = []
	resultG = []
	resultB = []
	for i in range(0, 256):
		resultR.append(0)
		resultG.append(0)
		resultB.append(0)

	for x in range(0,image.shape[0]):
		for y in range(0,image.shape[1]):
			(b, g, r) = image[x, y]

			resultR[r]+= 1
			resultG[g]+= 1
			resultB[b]+= 1
	
	result = [resultR, resultG, resultB]
	return result

def eqHist(hist, image):
	numPix = image.shape[0] * image.shape[1]
	histEqR = []
	histEqG = []
	histEqB = []

	#Calcula a probabilidade
	for i in range(0, 256):
		histEqR.append(hist[0][i]/numPix) #hist[0] aramzena resultR(ver hist() )
		histEqG.append(hist[1][i]/numPix) #hist[1] aramzena resultG(ver hist() )
		histEqB.append(hist[2][i]/numPix) #hist[2] aramzena resultB(ver hist() )

	#calcula a probabilidade acumulada
	for i in range(1, 256):
		histEqR[i] = histEqR[i]+histEqR[i-1]
		histEqG[i] = histEqG[i]+histEqG[i-1]
		histEqB[i] = histEqB[i]+histEqB[i-1]

	#equaliza o Histograma
	for i in range(0,256):
		histEqR[i] = histEqR[i] * 255
		histEqG[i] = histEqG[i] * 255
		histEqB[i] = histEqB[i] * 255

	histEq = [histEqR, histEqG, histEqB]
	return histEq

def aplicarHistEq(image, histEqualizado):
	result = deepcopy(image)

	for x in range(0,image.shape[0]):
		for y in range(0,image.shape[1]):
			(b, g, r) = image[x, y]
			resultR = histEqualizado[0][r]
			resultG = histEqualizado[0][g]
			resultB = histEqualizado[0][b]
			result[x, y] = (resultB, resultG, resultR)
	return result
########### FINAL DAS FUNCOES DE HISTOGRAMA #############