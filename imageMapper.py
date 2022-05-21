import cv2
import copy
import numpy as np
import imageGeometry as gm
import dataProcesser as datproc


class Mapper:
    def __init__(self,imageList_,dataMatrix_):
        '''
        ===============================================================================================
        Функция, запускаемая при инициализации класса.
        ===============================================================================================
        Принимает : imageList_  (Тип: ndArray) - список всех изображений.
        Принимает : dataMatrix_ (Тип: ndArray) - матрица, содержащая числовые данные о положении дрона.
        ===============================================================================================
        '''
        self.imageList = []
        self.dataMatrix = dataMatrix_
        # Процент сжатия изображений
        # для более точгного результата рекомендуется указать значение в 50%
        # уменьшение разрешения снимка для ускорения работы программы (30% - минимальное для удовлетворительного результата)
        scale_percent = 30
        # Расчет разрешения изображений, при условии если все фотографии с одного дрона (фотографии имеют одно соотношение сторон и разрешение)
        # Если изображения будут иметь слишком низкое разрешение 
        width = int(imageList_[0].shape[1] * scale_percent / 100)
        height = int(imageList_[0].shape[0] * scale_percent / 100)
        dsize = (width, height)
        # обработка изображений
        for i in range(0,len(imageList_)):
            width = int(imageList_[1].shape[1] * 30 / 100)
            height = int(imageList_[1].shape[0] * 30 / 100)
            image = imageList_[i]
            image = cv2.resize(image, dsize)
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
        ===============================================================================================
        Функция для наложения друг на друга двух фотоснимков.
        ===============================================================================================
        Принимает : index2 - индекс self.imageList и self.kpList для объединения с self.referenceImage 
            и self.referenceKeypoints
        ===============================================================================================
        Возвращает: result - комбинация двух изображений
        ===============================================================================================
        '''
        # Попытка объединить одну пару изображений на каждом шаге. 
        # Предпологается, что порядок, в котором даны изображения, является наилучшим порядком.
        image1 = copy.copy(self.imageList[index2 - 1])
        image2 = copy.copy(self.imageList[index2])

        '''
        Нахождение ключевых точек и вычисление дескриптора.
        '''
        detector = cv2.SIFT_create() # Альтернативой может послужить метод BRISK_create()
        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        ret1, mask1 = cv2.threshold(gray1,1,255,cv2.THRESH_BINARY)
        kp1, descriptors1 = detector.detectAndCompute(image1, mask1)

        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        ret2, mask2 = cv2.threshold(gray2,1,255,cv2.THRESH_BINARY)
        kp2, descriptors2 = detector.detectAndCompute(image2, mask2)

        '''
        Визуализация процедуры сопоставления
        '''
        keypoints1Im = cv2.drawKeypoints(image1,kp1,outImage = None,color=(0,100,255))
        datproc.display("Ключевые точки",keypoints1Im)
        keypoints2Im = cv2.drawKeypoints(image2,kp2,outImage = None,color=(0,100,255))
        datproc.display("Ключевые точки",keypoints2Im)

        matcher = cv2.BFMatcher() # использование грубого сопоставление
        matches = matcher.knnMatch(descriptors2,descriptors1, k=2) # нахождение пары ближайших совпадений
        # обрезка плохих совпадений
        good = []
        for m,n in matches:
            if m.distance < 0.55 * n.distance:
                good.append(m)
        matches = copy.copy(good)

        # Визуализация совпадений
        matchDrawing = datproc.drawMatches(gray2,kp2,gray1,kp1,matches)
        datproc.display("Совпадения",matchDrawing)
        # Сохранение изображения в папку "matches/"
        # cv2.imwrite("matches/match"+str(index2)+".png",matchDrawing) 

        # Синтаксис NumPy для извлечения данных о местоположении из структуры данных соответствия в матричной форме
        src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        '''
        Выполнение аффинного преобразования (Affine Transform)
        Аффинного преобразования должно хватить для выравнивания изображений.
        '''
        # Выполнение полной гомографии https://waksoft.susu.ru/2020/03/26/primery-gomogrfii-s-ispolzovaniem-opencv/
        A = cv2.estimateAffinePartial2D(src_pts,dst_pts)[0]
        if A.any() == None: 
            HomogResult = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
            H = HomogResult[0]

        '''
        Вычисление местоположения углов изображения.
        Тот же процесс, что и для warpPerspectiveWithPadding(), только за исключением того, что должны учитываться размеры двух изображений.
        '''
        height1,width1 = image1.shape[:2]
        height2,width2 = image2.shape[:2]
        corners1 = np.float32(([0,0],[0,height1],[width1,height1],[width1,0]))
        corners2 = np.float32(([0,0],[0,height2],[width2,height2],[width2,0]))
        warpedCorners2 = np.zeros((4,2)) # zeros() возвращает новый массив указанной формы и типа, заполненный нулями

        for i in range(0,4):
            cornerX = corners2[i,0]
            cornerY = corners2[i,1]
            if A.any() != None: # аффинное или перспективное преобразование
                warpedCorners2[i,0] = A[0,0]*cornerX + A[0,1]*cornerY + A[0,2]
                warpedCorners2[i,1] = A[1,0]*cornerX + A[1,1]*cornerY + A[1,2]
            else:
                warpedCorners2[i,0] = (H[0,0]*cornerX + H[0,1]*cornerY + H[0,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
                warpedCorners2[i,1] = (H[1,0]*cornerX + H[1,1]*cornerY + H[1,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
        allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
        [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
        [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)
            
        allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
        [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
        [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

        '''
        Выполнение выравнивания изображения и выравнивание по ключевым точкам
        '''
        translation = np.float32(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))
        warpedResImg = cv2.warpPerspective(self.resultImage, translation, (xMax-xMin, yMax-yMin))
        if A.any() == None:
            fullTransformation = np.dot(translation,H) #again, images must be translated to be 100% visible in new canvas
            warpedImage2 = cv2.warpPerspective(image2, fullTransformation, (xMax-xMin, yMax-yMin))
        else:
            warpedImageTemp = cv2.warpPerspective(image2, translation, (xMax-xMin, yMax-yMin))
            warpedImage2 = cv2.warpAffine(warpedImageTemp, A, (xMax-xMin, yMax-yMin))
            mask2 = cv2.threshold(warpedImage2, 0, 255, cv2.THRESH_BINARY)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, kernel)
            warpedImage2[mask2==0] = 0

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

        # отображение и сохранение результата
        result = warpedResImg + warpedImage2
        self.resultImage = result
        datproc.display("orthophoto",result)
        cv2.imwrite("orthophoto"+str(index2)+".png",result)
        
        return result