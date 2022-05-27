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
import ImageMapperDynamic


# Выбор метода
method = 0


# Выполнение накидного монтажа для двух изображений
if method == 0:
    # shutil.copyfile('data/2020_07_03_PhotoCamera_g401b40179_f001_028.JPG', 'res.JPG')
    img1 = "data/2020_07_03_PhotoCamera_g401b40179_f001_030.JPG"
    img2 = "res.JPG"
    ImageMapperDynamic.stitchImages(img1, img2)


# Выполнение накидного монтажа старым методом для всех снимков сразу
if method == 1:
    fileName = "data/telemetry-comma.txt" 
    imageDirectory = "data/" # Примеры фотографий https://drive.google.com/drive/folders/1bK5iiO7Ioe75V111r9WF0tYe4PayuW1C

    allImages, dataMatrix = dataProcesser.importData(fileName, imageDirectory)
    myCombiner = imageMapper.Mapper(allImages, dataMatrix)
    result = myCombiner.createMosaic()

    dataProcesser.display("Накидной монтаж", cv2.flip(result, 0)) # Зеркально оси X - вывод в исходном виде
    cv2.imwrite("orthophoto-result.png", cv2.flip(result, 0)) # Зеркально оси X - сохранение в исходном виде
    print("[INFO] Монтаж выполнен.")