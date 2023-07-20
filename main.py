import cv2
import numpy as np
import threading

from utils import *


def main(video_path, contours, warning_level=200, is_visualized=False):
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

        pool = []
        for contour_index in contours.keys():
            pool.append(
                threading.Thread(
                    target = check, 
                    args = (frame, contours, contour_index, warning_level, is_visualized)
                )
            )
        for t in pool:
            t.start()
        for t in pool:
            t.join()

        if is_visualized:
            for contour_index, val in contours.items():
                contour = val['contour']
                test_pos_x = contour[:, 0].min()
                test_pos_y = contour[:, 1].max()
                text = f'{contour_index} is warning: {val["result"]["state"]}'
                cv2.putText(
                    frame, text, (test_pos_x, test_pos_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2
                )
                text = ' | '.join([f'{k}: {round(v, 1)}' for k, v in val['result']['values'].items()])
                cv2.putText(
                    frame, text, (test_pos_x, test_pos_y+60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2
                )
                cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
                cv2.imshow('Image', frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    contours = [
        np.array([[685, 645], [980, 610], [980, 680], [710, 710]]),
        # np.array([[685, 645], [980, 610], [980, 680], [710, 710]]),
        # np.array([[685, 645], [980, 610], [980, 680], [710, 710]]),
    ]
    contours = {i: {'contour': cnt} for i, cnt in enumerate(contours)}
    main(
        video_path = '2023-07-12_03_49_11_03_55_00_4306a3db50ad28fc.mp4',
        contours = contours,
        is_visualized = True,
    )

    # TODO:
    # 1) разбить на ячейки
    # 2) определить значимые переменные по каждой ячейки
    # 3) вычислить результат среднего/отклонения/медианы по ячейкам
    # 4) логика заполнения линии снизу в вверх, иначе вывод warning
    # *) тест по двум параметрам яркости и красочности в формате HUE
    # *) определение выбросов в пункте (3)

