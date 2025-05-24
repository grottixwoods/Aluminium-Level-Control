import cv2
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from utils import check

@dataclass
class ContourConfig:
    """Конфигурация контура.
    
    Attributes:
        contour: Массив координат контура в формате numpy array
        bounds: Границы допустимых значений для статистических показателей
        warnings: Пороговые значения для предупреждений
        result: Результаты анализа контура, включая флаги и значения
    """
    contour: np.ndarray
    bounds: Dict[str, Tuple[float, float]]
    warnings: Dict[str, float]
    result: Optional[Dict[str, Any]] = None

def main(
    video_path: str,
    contours: List[ContourConfig],
    visualize: bool = True,
    video_save_path: Optional[str] = None
) -> None:
    """Основная функция для анализа уровня алюминия в ваннах по видео.
    
    Функция обрабатывает видеофайл, анализируя уровень алюминия в заданных контурах.
    Результаты анализа визуализируются на кадре и могут быть сохранены в новый видеофайл.
    В дальнейшем в зависимости от заказчика будет переделан в обработку реального времени.
    
    Args:
        video_path: Путь к видеофайлу для анализа
        contours: Список конфигураций контуров для анализа
        visualize: Флаг для отображения обработанного кадра в реальном времени
        video_save_path: Путь для сохранения результирующего видео
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    for contour in contours:
        contour.result = {
            'flags': {'is_warning': False},
            'values': {'mean': 0.0, 'std': 0.0, 'median': 0.0},
            'values_cell': {
                'all_mean': [],
                'all_std': [],
                'all_median': [],
                'all_coords': []
            },
            'flags_cell': {'is_warning_cell': []}
        }

    frame_per_second = cap.get(cv2.CAP_PROP_FPS)
    
    if video_save_path:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_shape = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(video_save_path, fourcc, frame_per_second, out_shape, True)
    
    frame_cur = 0
    frame_step = frame_per_second

    past_rows_colors: List[List[List[Optional[Tuple[int, int, int]]]]] = [[], [], []]
    final_colors: List[List[Tuple[int, int, int]]] = [[], [], []]

    while cap.isOpened():
        is_next, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cur)
        frame_cur += frame_step
        if not is_next:
            break

        pool = [
            threading.Thread(
                target=check,
                args=(frame.copy(), contours, contour_index)
            )
            for contour_index in range(len(contours))
        ]
        
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        
        if visualize or video_save_path:
            _visualize_frame(frame, contours, past_rows_colors, final_colors)
            
            if visualize:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if video_save_path:
                out.write(frame)

    cap.release()
    if video_save_path:
        out.release()
    cv2.destroyAllWindows()

def _visualize_frame(
    frame: np.ndarray,
    contours: List[ContourConfig],
    past_rows_colors: List[List[List[Optional[Tuple[int, int, int]]]]],
    final_colors: List[List[Tuple[int, int, int]]]
) -> None:
    """Визуализирует результаты анализа на кадре.
    
    Функция отображает контуры, их статистику и цветовую индикацию
    уровня заполнения для каждого региона.
    
    Args:
        frame: Кадр для визуализации
        contours: Список конфигураций контуров
        past_rows_colors: История цветов для каждого контура
        final_colors: Финальные цвета для каждого контура
    """
    for contour_index, contour in enumerate(contours):
        flags = contour.result['flags']
        flags_cell = contour.result['flags_cell']['is_warning_cell']
        values = contour.result['values']
        
        color = (0, 0, 255) if any(flags.values()) else (0, 255, 0)
        test_pos_x = contour.contour[:, 0].min()
        test_pos_y = contour.contour[:, 1].max()

        text = f'{contour_index} | ' + ' | '.join([f'{k}: {v}' for k, v in flags.items()])
        cv2.putText(frame, text, (test_pos_x, test_pos_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        text = ' | '.join([f'{k}: {round(v, 4)}' for k, v in values.items()])
        cv2.putText(frame, text, (test_pos_x, test_pos_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.drawContours(frame, [contour.contour], 0, color, 2)

        height = np.max(contour.contour[:, 1]) - np.min(contour.contour[:, 1])
        height_per_region = height // 4
        regions = _create_regions(contour.contour, height_per_region)
        
        rows_flags = np.array_split(flags_cell, 4)
        colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
        color_rows = [None, None, None, None]
        
        _process_regions(
            frame, regions, rows_flags, colors, color_rows,
            past_rows_colors, final_colors, contour_index
        )

def _create_regions(
    contour: np.ndarray,
    height_per_region: int
) -> List[np.ndarray]:
    """Создает регионы для анализа в контуре.
    
    Разбивает контур на 4 равных по высоте региона для анализа
    уровня заполнения.
    
    Args:
        contour: Массив координат контура
        height_per_region: Высота одного региона в пикселях
        
    Returns:
        Список массивов координат регионов
    """
    regions = []
    for i in range(4):
        y_min = np.min(contour[:, 1]) + i * height_per_region
        y_max = np.min(contour[:, 1]) + (i + 1) * height_per_region
        region_coordinates = np.array([
            [np.min(contour[:, 0]), y_min],
            [np.max(contour[:, 0]), y_min],
            [np.max(contour[:, 0]), y_max],
            [np.min(contour[:, 0]), y_max]
        ])
        regions.append(region_coordinates)
    return regions

def _process_regions(
    frame: np.ndarray,
    regions: List[np.ndarray],
    rows_flags: List[np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    color_rows: List[Optional[Tuple[int, int, int]]],
    past_rows_colors: List[List[List[Optional[Tuple[int, int, int]]]]],
    final_colors: List[List[Tuple[int, int, int]]],
    contour_index: int
) -> None:
    """Обрабатывает регионы контура и обновляет их визуализацию.
    
    Анализирует флаги для каждого региона и определяет их цвет
    на основе статистики заполнения.
    
    Args:
        frame: Кадр для визуализации
        regions: Список регионов контура
        rows_flags: Флаги для каждой строки
        colors: Словарь цветов для визуализации
        color_rows: Текущие цвета строк
        past_rows_colors: История цветов
        final_colors: Финальные цвета
        contour_index: Индекс контура
    """
    for i, region in enumerate(regions):
        region += np.array([[100, 200], [100, 200], [100, 200], [100, 200]])
        percentage = sum(rows_flags[i]) / len(rows_flags[i])

        if percentage >= 0.6:
            for j, color_row in enumerate(color_rows):
                if j >= i:
                    color_rows[j] = colors['red']
        elif color_rows[i] != colors['red']:
            color_rows[i] = colors['green']
        
        if i == 3:
            _update_color_history(
                past_rows_colors, final_colors,
                color_rows, contour_index, colors
            )

        if final_colors[contour_index]:
            cv2.fillPoly(frame, [region], color=final_colors[contour_index][i])
            cv2.drawContours(frame, [region], 0, (255, 255, 255), 2)

def _update_color_history(
    past_rows_colors: List[List[List[Optional[Tuple[int, int, int]]]]],
    final_colors: List[List[Tuple[int, int, int]]],
    color_rows: List[Optional[Tuple[int, int, int]]],
    contour_index: int,
    colors: Dict[str, Tuple[int, int, int]]
) -> None:
    """Обновляет историю цветов для контура.
    
    Поддерживает историю цветов за последние 10 кадров и определяет
    финальный цвет региона на основе преобладающего цвета в истории.
    
    Args:
        past_rows_colors: История цветов
        final_colors: Финальные цвета
        color_rows: Текущие цвета строк
        contour_index: Индекс контура
        colors: Словарь цветов
    """
    past_rows_colors[contour_index].append(color_rows)
    if len(past_rows_colors[contour_index]) == 10:
        past_rows_colors[contour_index].pop(0)
        red_counts = [0, 0, 0, 0]
        green_counts = [0, 0, 0, 0]
        
        for sublist in past_rows_colors[contour_index]:
            for index, item in enumerate(sublist):
                if item == colors['red']:
                    red_counts[index] += 1
                elif item == colors['green']:
                    green_counts[index] += 1

        final_colors[contour_index] = [
            colors['red'] if r > g else colors['green']
            for r, g in zip(red_counts, green_counts)
        ][-4:]

if __name__ == '__main__':
    contours = [
        ContourConfig(
            contour=np.array([[1155, 660], [1412, 625], [1412, 677], [1162, 715]]),
            bounds={
                'mean': (200, 235),
                'median': (200, 235),
                'std': (0, 105),
            },
            warnings={
                'mean': 0.55,
                'median': 0.55,
                'std': 1.05,
            },
        ),
        ContourConfig(
            contour=np.array([[685, 645], [980, 610], [980, 680], [710, 710]]),
            bounds={
                'mean': (200, 230),
                'median': (200, 230),
                'std': (0, 100),
            },
            warnings={
                'mean': 0.6,
                'median': 0.6,
                'std': 1.1,
            },
        ),
        ContourConfig(
            contour=np.array([[262, 407], [492, 368], [497, 400], [267, 435]]),
            bounds={
                'mean': (140, 200),
                'median': (140, 200),
                'std': (0, 110),
            },
            warnings={
                'mean': 1.6,
                'median': 0.75,
                'std': 1.8,
            },
        ),
    ]
    
    main(
        video_path='2023-07-12_03_49_11_03_55_00_4306a3db50ad28fc.mp4',
        contours=contours,
        visualize=True,
        video_save_path='output.mp4',
    )
