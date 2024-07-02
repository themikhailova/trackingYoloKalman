import datetime
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import numpy as np

# Порог уверенности для фильтрации
CONFIDENCE_THRESHOLD = 0.35


GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

video_cap = cv2.VideoCapture("video_1.mp4")

# Загрузка предобученной модели YOLO
model = YOLO("yolov8n.pt")

# Словарь для хранения активных треков
tracks = {}

# Инициализация фильтров Калмана для каждого трека
kalman_filters = {}

# Максимальное количество пропущенных кадров перед удалением трека
MAX_CONSECUTIVE_INVISIBLE_COUNT = 5


# Функция для инициализации фильтра Калмана
def init_kalman_filter(x, y):
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Матрица перехода состояний
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # Матрица измерений
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # Начальная оценка ковариации ошибки состояния
    kf.P *= 10
    # Ковариация шума измерений
    kf.R = 5
    # Ковариация шума процесса (дискретный белый шум)
    kf.Q = np.eye(4) * 1e-2
    # Начальное состояние: координаты центра и скорости
    kf.x = np.array([[x],
                     [y],
                     [0.],
                     [0.]])
    return kf


# Выходное видео
def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Заменено 'MP4V' на 'mp4v'
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


writer = create_video_writer(video_cap, "output_2test_12345678.mp4")

while True:
    start = datetime.datetime.now()
    ret, frame = video_cap.read()

    if not ret:
        break

    # Запуск модели YOLO на кадре
    detections = model(frame)[0]

    # Список для хранения результатов детекции
    results = []

    # Обработка детекций
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if data[5] == 2.0:
            # Фильтрация слабых детекций
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # Извлечение координат bbox'а
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

            # Добавление bbox'а и уверенности в список результатов
            results.append({"bbox": [xmin, ymin, xmax - xmin, ymax - ymin], "confidence": confidence})

    # Обновление треков
    for track_id in list(tracks.keys()):
        if track_id not in [i for i in range(len(results))]:
            tracks[track_id]["consecutiveInvisibleCount"] += 1
            if tracks[track_id]["consecutiveInvisibleCount"] > MAX_CONSECUTIVE_INVISIBLE_COUNT:
                del tracks[track_id]
                del kalman_filters[track_id]
        else:
            tracks[track_id]["consecutiveInvisibleCount"] = 0

    # Обработка результатов детекции и обновление треков
    for i, data in enumerate(results):
        bbox = data["bbox"]
        confidence = data["confidence"]
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        center = (int((xmin + xmax) / 2.0), int((ymin + ymax) / 2.0))

        # Поиск ближайшего трека к текущей детекции
        if len(tracks) > 0:
            closest_track_id = min(tracks, key=lambda tid: cv2.norm(center, tracks[tid]["prediction"]))
            min_distance = cv2.norm(center, tracks[closest_track_id]["prediction"])

            if min_distance < 50:
                track_id = closest_track_id
                tracks[track_id]["bbox"] = bbox
                tracks[track_id]["prediction"] = center
                tracks[track_id]["age"] += 1
                tracks[track_id]["totalVisibleCount"] += 1
                tracks[track_id]["consecutiveInvisibleCount"] = 0

                # Обновление фильтра Калмана новыми координатами центра, если есть предсказание
                if kalman_filters[track_id] is not None:
                    prediction = kalman_filters[track_id].predict()
                    if prediction is not None:
                        tracks[track_id]["prediction"] = (prediction[0], prediction[1])

                # Обновление фильтра Калмана новыми координатами центра
                kalman_filters[track_id].update(np.array([[center[0]], [center[1]]]))

            else:
                new_track_id = max(tracks.keys()) + 1 if len(tracks) > 0 else 0
                tracks[new_track_id] = {
                    "bbox": bbox,
                    "prediction": center,
                    "age": 1,
                    "totalVisibleCount": 1,
                    "consecutiveInvisibleCount": 0
                }
                kalman_filters[new_track_id] = init_kalman_filter(center[0], center[1])

        else:
            track_id = i
            tracks[track_id] = {
                "bbox": bbox,
                "prediction": center,
                "age": 1,
                "totalVisibleCount": 1,
                "consecutiveInvisibleCount": 0
            }
            kalman_filters[track_id] = init_kalman_filter(center[0], center[1])

    # Отображение активных треков и ограничивающих прямоугольников
    for track_id in list(tracks.keys()):
        if tracks[track_id]["totalVisibleCount"] > 0:
            bbox = tracks[track_id]["bbox"]
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            predicted_center = kalman_filters[track_id].x[:2].reshape(2).astype(int)
            cv2.circle(frame, tuple(predicted_center), 4, (255, 0, 0), -1)

    end = datetime.datetime.now()
    processing_time = (end - start).total_seconds()

    # FPS на кадре
    fps = f"FPS: {1 / processing_time:.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    cv2.imshow("Кадр", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Освобождение ресурсов
video_cap.release()
writer.release()
cv2.destroyAllWindows()
