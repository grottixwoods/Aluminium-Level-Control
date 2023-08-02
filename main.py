import cv2
import numpy as np
import threading

from utils import *


def main(video_path, contours, is_visualized=False, video_save_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_per_second = cap.get(cv2.CAP_PROP_FPS)

    if video_save_path:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_shape = int(cap.get(3)), int(cap.get(4)) # width, height
        out = cv2.VideoWriter(video_save_path, fourcc, frame_per_second, out_shape, True)
    
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
                    args = (frame, contours, contour_index)
                )
            )
        for t in pool:
            t.start()
        for t in pool:
            t.join()

        if is_visualized or video_save_path:
            for contour_index, val in contours.items():
                contour = val['contour']
                flags = val['result']['flags']
                flags_cell = val['result']['flags_cell']['is_warning_cell']
                values = val['result']['values']
                values_cell = val['result']['values_cell']
                color = (0, 0, 255) if any(flags.values()) else (0, 255, 0)
                test_pos_x = contour[:, 0].min()
                test_pos_y = contour[:, 1].max()
                text = f'{contour_index} | ' + ' | '.join([f'{k}: {v}' for k, v in flags.items()])
                cv2.putText(
                    frame, text, (test_pos_x, test_pos_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2
                )
                text = ' | '.join([f'{k}: {round(v, 4)}' for k, v in values.items()])
                cv2.putText(
                    frame, text, (test_pos_x, test_pos_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2
                )
                # Вывод рамки в резервуаре
                cv2.drawContours(frame, [contour], 0, color, 2)
                # Разбитие рамки на 4 уровня
                height = np.max(contour[:, 1]) - np.min(contour[:, 1])
                height_per_region = height // 4
                regions = []
                for i in range(4):
                    y_min = np.min(contour[:, 1]) + i * height_per_region
                    y_max = np.min(contour[:, 1]) + (i + 1) * height_per_region
                    region_coordinates = np.array([[np.min(contour[:, 0]), y_min],
                                                   [np.max(contour[:, 0]), y_min],
                                                   [np.max(contour[:, 0]), y_max],
                                                   [np.min(contour[:, 0]), y_max]])
                    regions.append(region_coordinates)
                # Разбитие списка warnings каждой ячейки на 4 уровня (на 4 списка)
                rows_flags = np.array_split(flags_cell, 4)
                # Отрисовка уровней, окрашивая их в зависимости от количества ячеек с warning
                colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
                color_rows = [None, None, None, None]
                for i, region in enumerate(regions):
                    region += np.array([[100, 200], [100, 200], [100, 200], [100, 200]])
                    percentage = sum(rows_flags[i]) / len(rows_flags[i])
                    if percentage >= 0.5:
                        for j, color_row in enumerate(color_rows):
                            if j >= i:
                                color_rows[j] = colors['red']

                    # elif sum(rows_flags[i]) / len(rows_flags[i]) >= 0.4:
                    #     color_row = (0, 255, 255)
                    elif color_rows[i] != colors['red']:
                        color_rows[i] = colors['green']

                    cv2.fillPoly(frame, [region], color=color_rows[i])
                    cv2.drawContours(frame, [region], 0, (255, 255, 255), 2)

            if is_visualized:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if video_save_path:
                out.write(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    contours = [
        {
            'contour': np.array([[1155, 660], [1412, 625], [1412, 677], [1162, 715]]),
            'bounds': {
                'mean': (200, 230),
                'median': (200, 230),
                'std': (0, 100),
            },
            'warnings': {
                'mean': 0.5,
                'median': 0.5,
                'std': 1.0,
            },
        },
        {
            'contour': np.array([[685, 645], [980, 610], [980, 680], [710, 710]]),
            'bounds': {
                'mean': (200, 230),
                'median': (200, 230),
                'std': (0, 100),
            },
            'warnings': {
                'mean': 0.5,
                'median': 0.5,
                'std': 1.0,
            },
        },
        {
            'contour': np.array([[262, 405], [492, 365], [497, 400], [267, 435]]),
            'bounds': {
                'mean': (140, 190),
                'median': (140, 190),
                'std': (0, 100),
            },
            'warnings': {
                'mean': 1.5,
                'median': 0.6,
                'std': 1.7,
            },
        },
    ]
    contours = {i: v for i, v in enumerate(contours)}
    main(
        video_path='2023-07-12_03:49:11_03:55:00_4306a3db50ad28fc.mp4',
        contours=contours,
        is_visualized=True,
        video_save_path='out_2023-07-12_03:49:11_03:55:00_4306a3db50ad28fc.mp4',
    )

    # TODO:
    # 1) Подправить пороговые значения main() / подобрать оптимальные значения каллибровки в utils
    # 2) измнение флага is_outlier
    # 3) тест по двум параметрам яркости и красочности в формате HUE

