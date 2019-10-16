# Importação das bibliotecas
import cv2
import numpy as np
from copy import deepcopy
from functions import *

# Leitura da imagem com a função imread()
imagem = cv2.imread('images/cap6/Fig0630(01)(strawberries_fullcolor).TIF')
imgNeg = negativeFilter(imagem)
imgLog = logFilter(imagem, 26)

cv2.namedWindow("OriginalPic", cv2.WINDOW_NORMAL)
cv2.imshow("OriginalPic", imagem)

#cv2.namedWindow("DisplayWindow", cv2.WINDOW_NORMAL)
#cv2.imshow("DisplayWindow", imgLog)

histograma = hist(imagem)
#print(histograma[2])

histEq = eqHist(histograma, imagem) #equaliza o histograma
#print(histEq[2])

#imgEq = aplicarHistEq(imagem, histEq) #aplica o histograma equalizado

#imgGamma = gammaFilter(imagem, 2, 0.8)
#imgGauss = gaussianFilter(imagem, 3, 3)

convMat = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 0]])
convImg = convFilter(imagem, convMat, 3)

cv2.namedWindow("DisplayWindow", cv2.WINDOW_NORMAL)
cv2.imshow("DisplayWindow", convImg)

cv2.waitKey(0) #espera pressionar qualquer tecla


