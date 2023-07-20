import cv2
import numpy as np

from utils import *


def main(video_path, contour, warning_level=200, is_visualized=False):
    text_pos = (685, 740) # contour.min(0), contour.max(1) + 30

    cap = cv2.VideoCapture(video_path)
    frame_per_second = cap.get(cv2.CAP_PROP_FPS)

    frame_cur = 0
    frame_step = frame_per_second
    while cap.isOpened():
        is_next, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cur)
        frame_cur += frame_step
        if not is_next:
            break

        image = frame # copy?
        rotated = rotate(image, contour)
        # cv2.imshow('Rotated', rotated)
        # cells = cell_split(rotated, rows, cols)

        mean_color = rotated.mean()
        if mean_color >= warning_level:
            warning(warning_level)

        if is_visualized:
            cv2.putText(
                image,
                f's: {frame_cur//frame_per_second} | mean_color: {round(mean_color, 2)}',
                text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 1)
            cv2.imshow('Image', image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()


if __name__ == '__main__':
    main(
        video_path = '2023-07-12_03_49_11_03_55_00_4306a3db50ad28fc.mp4',
        contour = np.array([[685, 645], [980, 610], [980, 680], [710, 710]]),
        is_visualized = True,
    )

    # TODO:
    # 1) разбить на ячейки
    # 2) определить значимые переменные по каждой ячейки
    # 3) вычислить результат среднего/отклонения/медианы по ячейкам
    # 4) логика заполнения линии снизу в вверх, иначе вывод warning
    # *) тест по двум параметрам яркости и красочности в формате HUE
    # *) определение выбросов в пункте (3)

