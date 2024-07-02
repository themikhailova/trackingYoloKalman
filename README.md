﻿# trackingYoloKalman



https://github.com/themikhailova/trackingYoloKalman/assets/91223359/d0eeab5d-0055-448b-876a-443801ce601a


Для трекера используются детектор Yolo в двух вариантах моделей (yolov5s и yolov8n)  
Проверка проводилась на двух видеорядах. Результаты запусков представлены ниже:  

|     Видео      | Количество кадров | Количество кадров с треками  | Время жизни трека (в кадрах) | Модель | Время обработки, сек| Среднее fps |
|     :---:      |       :---:       |             :---:            |             :---:            |  :---: |        :---:        |    :---:    |
|       1        |        1226       |              875             |               23             |    8   |        263.94       |    4.65     |
|       1        |        1226       |              933             |               63             |    5   |        848.37       |    1.45     |
|       2        |        7467       |              5746            |              1.5             |    8   |        1382.75      |    5.40     |
|       2        |        7467       |              4421            |                2             |    5   |        4155.52      |    1.80     |

Также, используется фильтр Калмана. Фильтр Калмана обеспечивает более стабильное и плавное отслеживание объектов, предсказывая следующее положение объекта на основе предыдущих наблюдений. Когда детекция отсутствует, фильтр Калмана может предсказать положение объекта на основе его предыдущего движения. Фильтр Калмана помогает ассоциировать новые детекции с существующими треками. Предсказанное положение объекта используется для поиска ближайшей детекции, что позволяет правильно обновить трек.   

Алгоритм:
1. Каждый кадр проходит детекцию с помощью модели YOLO
2. Цикл обработки детекций:  
   2.1 Для каждого bbox'а проверка, машина ли найдена и проходит ли детекция порог уверенности  
   2.2 Если да, то сохраняем результат  
3. Цикл по всем уже имеющимся трекам:  
   3.1 Если имеющихся треков нет в новых найденных треках:  
   - 3.1.1 Увеличивается счетчик пропущенных треков  
   - 3.1.2 Если трек пропущен более 5 раз, он удаляется  
4. По каждому результату детекции:  
   4.1 Если треки уже есть:  
      4.1.1 Для текущей детекции ищется ближайший трек с помощью евклидова расстояния от предсказанного центра текущего трека до текущей детекции  
      4.1.2 Если расстояние меньше 50 пикселей, то происходит обновление трека. Иначе: создается новый трек  


На Google Disk выложены взодные видео и реузьтаты работы на них обеих моделей: https://drive.google.com/drive/folders/1XoThU4ehMp9OaxCQj_fgaIjM9cGI3duS?usp=sharing  
