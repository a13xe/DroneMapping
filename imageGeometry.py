import cv2
import math as m
import numpy as np


def computeUnRotMatrix(pose):
    '''
    Удаление вращения дрона.
    Функция выполнена по формулам http://planning.cs.uiuc.edu/node102.html.
    Принимает  : pose: NumPy ndArray 1x6, содержащий информацию о положении дрона в формате [X,Y,Z,Y,P,R].
    Возвращает : Матрица вращения 3x3, удаляющая искажение перспективы из изображения, к которому она применяется.
    '''
    a = pose[3]*np.pi/180 # alpha
    b = pose[4]*np.pi/180 # beta
    g = pose[5]*np.pi/180 # gamma
    # Вычисление матрицы R.
    Rz = np.array(([m.cos(a)    , -1*m.sin(a)   , 0         ],
                   [m.sin(a)    , m.cos(a)      , 0         ],
                   [0           , 0             , 1         ]))
    Ry = np.array(([m.cos(b)    , 0             , m.sin(b)  ],
                   [0           , 1             , 0         ],
                   [-1*m.sin(b) , 0             , m.cos(b)  ]))
    Rx = np.array(([1           , 0             , 0         ],
                   [0           , m.cos(g)      ,-1*m.sin(g)],
                   [0           , m.sin(g)      , m.cos(g)  ]))
    Ryx = np.dot(Rx,Ry)

    R = np.dot(Rz,Ryx)
    R[0,2] = 0
    R[1,2] = 0
    R[2,2] = 1
    Rtrans = R.transpose()
    InvR = np.linalg.inv(Rtrans)

    #Возврат обратной матрицы R.
    return InvR


def warpPerspectiveWithPadding(image,transformation):
    '''
    Когда мы деформируем изображение, его углы могут оказаться за пределами исходного изображения. 
    Эта функция создает новый образ, который гарантирует, что этого не произойдет.
    Принимает  : image изображение ndArray
    Принимает  : transform 3x3 ndArray, представляющий трансформацию перспективы
    Принимает  : kp ключевые точки, связанные с изображением
    Возвращает : преобразованное изображение
    '''

    height = image.shape[0]
    width = image.shape[1]
    corners = np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2) # первоначальное положение углов

    warpedCorners = cv2.perspectiveTransform(corners, transformation) # положение с искривленными углами
    [xMin, yMin] = np.int32(warpedCorners.min(axis=0).ravel() - 0.5) # новые измерения
    [xMax, yMax] = np.int32(warpedCorners.max(axis=0).ravel() + 0.5)
    translation = np.array(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1])) # необходимо перевести изображение так, чтобы все было видно
    fullTransformation = np.dot(translation,transformation) # составить варп и трансляцию в правильном порядке
    result = cv2.warpPerspective(image, fullTransformation, (xMax-xMin, yMax-yMin))

    return result