import cv2
import dataProcesser as datproc


def stitchImages(img1, img2):
    '''
    ===============================================================================================
    Функция для объединения фотоснимков.
    В результате выполнения мы получаем ортофотографию, составленную из двух снимков, которую
    в дальнейшем также мажно послать в качестве входных данных в эту же функцию.
    ===============================================================================================
    Принимает : img1 - наименование файла первого изображения
    Принимает : img2 - наименование файла второго изображения
    ===============================================================================================
    '''
    # Получение  входного изображения и инициализация список изображений
    # В данном модуле не рекомендуется уменьшать разрешение снимков,
    # так как ортоизображение, составленное в предыдущей итерации, при последующем объединении с новыми снимками будет тоже менять разрешение.
    print("[INFO] Загрузка изображений ...")
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    # Инициализация объекта "сшивания" изображений OpenCV - затем выполнение накидного монтажа
    print("[INFO] Построение накидного монтажа ...")
    # Если OpenCV версии 4.x
    stitcher = cv2.Stitcher.create(mode = 1) # Mode { PANORAMA = 0,  SCANS = 1 } - атрибут, отвечающий за тип функции Stitcher()
    (status, stitched) = stitcher.stitch((image1,image2))

    # Если статус равен 0, то OpenCV успешно реализует сшивание изображений
    if status == 0:
        # Записываем сшитый образ на жесткий диск
        cv2.imwrite("res.JPG",stitched)
        # Выводим сшитое изображение на экран
        datproc.display("Накидной монтаж",stitched)
        print("[INFO] Монтаж выполнен.")
    else:
        # В противном случае склейка не удалась, возможно, из-за недостаточного количества обнаруженных ключевых точек
        # Status {
        #   OK = 0,
        #   ERR_NEED_MORE_IMGS = 1,
        #   ERR_HOMOGRAPHY_EST_FAIL = 2,
        #   ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
        # }
        print("[INFO] Монтаж не выполнен. Код ошибки: ({})".format(status))