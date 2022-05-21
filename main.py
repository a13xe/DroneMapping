'''
===============================================================================================
Модуль построения накидного монтажа.
===============================================================================================
Описание исходных переменных, задаваемых пользователем:
- fileName - переменная, содержащая в себе путь к файлу с телеметрическими данными фотоснимков;
- imageDirectory - директория, в которой хранятся фотоснимки;
===============================================================================================
Описание результирующих переменных:
- allImages   - масиссив изображений;
- dataMatrix  - массив телеметрических данных;
- myCombiner  - объект класса, в котором сопоставляются все изображения;
- result      - ортоизображение.
===============================================================================================
'''


import cv2
import imageMapper
import dataProcesser


fileName = "data/telemetry-comma.txt" 
imageDirectory = "data/" # Примеры фотографий https://drive.google.com/drive/folders/1bK5iiO7Ioe75V111r9WF0tYe4PayuW1C

allImages, dataMatrix = dataProcesser.importData(fileName, imageDirectory)
myCombiner = imageMapper.Mapper(allImages, dataMatrix)
result = myCombiner.createMosaic()

dataProcesser.display("Накидной монтаж", cv2.flip(result, 0)) # Зеркально оси X - вывод в исходном виде
cv2.imwrite("orthophoto-result.png", cv2.flip(result, 0)) # Зеркально оси X - сохранение в исходном виде