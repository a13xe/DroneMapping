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


method = 0 # Выбор метода
scale_percent = 100 # Процент сжатия изображений.


# Выполнение накидного монтажа для двух изображений
if method == 0:
    image_first_name = "data/2020_07_03_PhotoCamera_g401b40179_f001_025.JPG"
    image_first = cv2.imread(image_first_name)
    if scale_percent < 100:
        dsize = (int(image_first.shape[1] * scale_percent / 100), int(image_first.shape[0] * scale_percent / 100))
        image_first = cv2.resize(image_first, dsize)
    cv2.imwrite("res.JPG",image_first)
    # выполнение монтажа снимков по поступлению
    img1 = "data/2020_07_03_PhotoCamera_g401b40179_f001_026.JPG"
    img2 = "res.JPG"     
    ImageMapperDynamic.stitchImages(img1, img2, scale_percent)


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