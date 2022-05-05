import cv2
import copy
import numpy as np
import imageGeometry as gm
import dataProcesser as datproc
# import matplotlib.pyplot as plt


class Mapper:
    def __init__(self,imageList_,dataMatrix_):
        '''
        Принимает imageList_  : список всех изображений в наборе данных.
        Принимает dataMatrix_ : Матрица, содержащая числовые данные о положении дрона.
        '''
        self.imageList = []
        self.dataMatrix = dataMatrix_
        detector = cv2.ORB()
        for i in range(0,len(imageList_)):
            image = imageList_[i][::3,::3,:] # уменьшите разрешение изображения, чтобы ускорить процесс.
            M = gm.computeUnRotMatrix(self.dataMatrix[i,:])
            # Выполнение перспективного преобразования на основе информации о положении дрона.
            # В лечшем случае это сделает каждое изображение таким, как будто оно просматривается сверху (ортоизображение).
            # Предполагается, что плоскость заземления идеально плоская.
            correctedImage = gm.warpPerspectiveWithPadding(image,M)
            self.imageList.append(correctedImage) # хранение только подходящих изображений для использования
        self.resultImage = self.imageList[0]


    def createMosaic(self):
        for i in range(1,len(self.imageList)):
            self.combine(i)
        return self.resultImage


    def combine(self, index2):
        '''
        Принимает index2 : индекс self.imageList и self.kpList для объединения с self.referenceImage и self.referenceKeypoints
        Возвращает комбинацию двух изображений
        '''
        # Попытка объединить одну пару изображений на каждом шаге. 
        # Предпологается, что порядок, в котором даны изображения, является наилучшим порядком.
        image1 = copy.copy(self.imageList[index2 - 1])
        image2 = copy.copy(self.imageList[index2])

        '''
        Нахождение ключевых точек и вычисление дескриптора.
        '''
        detector = cv2.BRISK_create()
        # detector.extended = True
        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        ret1, mask1 = cv2.threshold(gray1,1,255,cv2.THRESH_BINARY)
        kp1, descriptors1 = detector.detectAndCompute(image1, None)

        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        ret2, mask2 = cv2.threshold(gray2,1,255,cv2.THRESH_BINARY)
        kp2, descriptors2 = detector.detectAndCompute(image2, None)

        '''
        Визуализация процедуры сопоставления
        '''
        keypoints1Im = cv2.drawKeypoints(image1,kp1,outImage = None,color=(0,0,255))
        datproc.display("KEYPOINTS",keypoints1Im)
        keypoints2Im = cv2.drawKeypoints(image2,kp2,outImage = None,color=(0,0,255))
        datproc.display("KEYPOINTS",keypoints2Im)

        matcher = cv2.BFMatcher() # использование грубого сопоставление
        matches = matcher.knnMatch(descriptors2,descriptors1, k=2) # нахождение пары ближайших совпадений
        # обрезка плохих совпадений
        good = []
        for m,n in matches:
            if m.distance < 0.55*n.distance:
                good.append(m)
        matches = copy.copy(good)

        # Визуализация совпадений
        matchDrawing = datproc.drawMatches(gray2,kp2,gray1,kp1,matches)
        datproc.display("matches",matchDrawing)

        # Синтаксис NumPy для извлечения данных о местоположении из структуры данных соответствия в матричной форме
        src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        '''
        Выполнение векторного преобразования (Affine Transform)
        Идея: поскольку мы исправили ориентацию камеры, векторного преобразования должно хватить для выравнивания изображений.
        '''
        # оценка RigidTransform - Вычисляет универсальное векторного преобразование между двумя наборами двумерных точек.
        fullAffine=False
        A = cv2.estimateAffinePartial2D(src_pts,dst_pts) # false, потому что нам нужно только 5 DOF.
                                                         # при вращении было уделено 3 DOF
        # if A == None: # Если RANSAC дал сбой в estimateAffinePartial2D(), 
                      # то выполняется попытка полной гомографии.
                      # https://waksoft.susu.ru/2020/03/26/primery-gomogrfii-s-ispolzovaniem-opencv/
        HomogResult = cv2.findHomography(src_pts,dst_pts,method=cv2.RANSAC)
        H = HomogResult[0]

        '''
        Вычислить 4 местоположения углов изображения
        Идея: Тот же процесс, что и для warpPerspectiveWithPadding(), за исключением
              того, что должны учитываться размеры двух изображений.
        '''
        height1,width1 = image1.shape[:2]
        height2,width2 = image2.shape[:2]
        corners1 = np.float32(([0,0],[0,height1],[width1,height1],[width1,0]))
        corners2 = np.float32(([0,0],[0,height2],[width2,height2],[width2,0]))
        # zeros() возвращает новый массив указанной формы и типа, заполненный нулями
        warpedCorners2 = np.zeros((4,2)) 

        for i in range(0,4):
            cornerX = corners2[i,0]
            cornerY = corners2[i,1]

            # if A != None: #check if we're working with affine transform or perspective transform
            #     warpedCorners2[i,0] = A[0,0]*cornerX + A[0,1]*cornerY + A[0,2]
            #     warpedCorners2[i,1] = A[1,0]*cornerX + A[1,1]*cornerY + A[1,2]
            # else:
            #     warpedCorners2[i,0] = (H[0,0]*cornerX + H[0,1]*cornerY + H[0,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
            #     warpedCorners2[i,1] = (H[1,0]*cornerX + H[1,1]*cornerY + H[1,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])

        allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
        [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
        [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

        '''
        Выполнение выравнивания изображения и выравнивание по ключевым точкам
        '''
        translation = np.float32(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))
        warpedResImg = cv2.warpPerspective(self.resultImage, translation, (xMax-xMin, yMax-yMin))
        
        # if A is None:
        fullTransformation = np.dot(translation,H) # изображения должны быть переведены, чтобы быть полностью видимыми на новом холсте.
        warpedImage2 = cv2.warpPerspective(image2, fullTransformation, (xMax-xMin, yMax-yMin))
        # else:
        #     warpedImageTemp = cv2.warpPerspective(image2, translation, (xMax-xMin, yMax-yMin))
        #     warpedImage2 = cv2.warpAffine(warpedImageTemp, A, (xMax-xMin, yMax-yMin)) # Векторная трансформация
        #     mask2 = cv2.threshold(warpedImage2, 0, 255, cv2.THRESH_BINARY)[1]
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        #     mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, kernel)
        #     warpedImage2[mask2==0] = 0

        self.imageList[index2] = copy.copy(warpedImage2) # обновление старых изображений для извлечения функций в будущем

        resGray = cv2.cvtColor(self.resultImage,cv2.COLOR_BGR2GRAY)
        warpedResGray = cv2.warpPerspective(resGray, translation, (xMax-xMin, yMax-yMin))

        '''
        Построение маски для комбинации изображений
        '''
        ret, mask1 = cv2.threshold(warpedResGray,1,255,cv2.THRESH_BINARY_INV)
        mask3 = np.float32(mask1)/255

        # применение маски
        warpedImage2[:,:,0] = warpedImage2[:,:,0]*mask3
        warpedImage2[:,:,1] = warpedImage2[:,:,1]*mask3
        warpedImage2[:,:,2] = warpedImage2[:,:,2]*mask3

        result = warpedResImg + warpedImage2
        # отображение и сохранение результата
        self.resultImage = result
        datproc.display("orthophoto",result)
        cv2.imwrite("orthophoto"+str(index2)+".png",result)
        
        return result