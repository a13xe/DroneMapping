'''
Построение накидного монтажа.
'''


import cv2
import imageMapper
import dataProcesser


fileName = "data/telemetry-comma.txt" # Фотографии (кроме первых двух) расположены https://drive.google.com/drive/folders/1bK5iiO7Ioe75V111r9WF0tYe4PayuW1C
imageDirectory = "data/"
allImages, dataMatrix = dataProcesser.importData(fileName, imageDirectory)
myCombiner = imageMapper.Mapper(allImages, dataMatrix)
result = myCombiner.createMosaic()

# dataProcesser.display("Накидной монтаж", result)
# cv2.imwrite("orthophoto-result.png", result)

dataProcesser.display("Накидной монтаж", cv2.flip(result, 0)) # Зеркально оси X - вывод в исходном виде
cv2.imwrite("orthophoto-result.png", cv2.flip(result, 0)) # Зеркально оси X - сохранение в исходном виде