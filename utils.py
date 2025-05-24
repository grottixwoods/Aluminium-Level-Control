import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from dataclasses import dataclass

@dataclass
class StatisticsResult:
    """Результаты статистического анализа региона изображения.
    
    Attributes:
        mean: Среднее значение яркости в регионе
        std: Стандартное отклонение яркости в регионе
        median: Медианное значение яркости в регионе
    """
    mean: float
    std: float
    median: float

def crop(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Обрезает изображение по указанным координатам.
    
    Создает маску по заданным координатам и обрезает изображение
    по границам этой маски.
    
    Args:
        img: Исходное изображение
        pts: Массив координат для обрезки
        
    Returns:
        Обрезанное изображение по указанным координатам
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(pts)
    cropped = res[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    return cropped

def order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает координаты четырехугольника.
    
    Координаты упорядочиваются в следующем порядке:
    - верх-лево
    - верх-право
    - низ-право
    - низ-лево
    
    Args:
        pts: Массив координат четырехугольника
        
    Returns:
        Упорядоченный массив координат
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def rotate(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Выполняет перспективное преобразование изображения.
    
    Преобразует изображение в прямоугольник с правильной перспективой
    на основе заданных координат четырехугольника.
    
    Args:
        img: Исходное изображение
        pts: Массив координат четырехугольника
        
    Returns:
        Изображение с правильной перспективой
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def calibrate_frame(
    frame: np.ndarray,
    target_mean: float,
    target_std: float
) -> np.ndarray:
    """Калибрует изображение, изменяя яркость и контраст.
    
    Приводит изображение к заданным значениям яркости и контраста
    для обеспечения стабильности измерений.
    
    Args:
        frame: Исходное изображение
        target_mean: Целевое среднее значение яркости
        target_std: Целевое стандартное отклонение яркости
        
    Returns:
        Калиброванное изображение
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray_frame)
    std_val = np.std(gray_frame)
    contrast = target_std / std_val
    brightness = target_mean - (mean_val * contrast)
    adjusted_frame = cv2.convertScaleAbs(gray_frame, alpha=contrast, beta=brightness)
    calibrated_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)
    return calibrated_frame

def quantization(
    img: np.ndarray,
    clusters: int = 8,
    rounds: int = 1
) -> np.ndarray:
    """Применяет квантование цветов изображения.
    
    Использует алгоритм k-средних для уменьшения количества цветов
    в изображении, что помогает снизить шум и упростить анализ.
    
    Args:
        img: Исходное изображение
        clusters: Количество кластеров для квантования
        rounds: Количество итераций k-средних
        
    Returns:
        Изображение после квантования
    """
    samples = img.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        samples, clusters, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
        rounds, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(img.shape)

def grid(
    img: np.ndarray,
    cells_in_height: int,
    cells_in_width: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """Разбивает изображение на ячейки заданного размера.
    
    Args:
        img: Исходное изображение
        cells_in_height: Количество ячеек в высоте
        cells_in_width: Количество ячеек в ширине
        
    Returns:
        Словарь, где ключи - координаты ячеек (y, x),
        значения - соответствующие части изображения
    """
    h, w = img.shape[:2]
    d_h = h // cells_in_height
    d_w = w // cells_in_width
    return {
        (i, j): img[i:i + d_h, j:j + d_w]
        for i in range(0, h, d_h)
        for j in range(0, w, d_w)
    }

def calculate_statistics_in_tiles(
    tiles: Dict[Tuple[int, int], np.ndarray]
) -> Dict[Tuple[int, int], StatisticsResult]:
    """Вычисляет статистику для каждой ячейки изображения.
    
    Args:
        tiles: Словарь с координатами и изображениями ячеек
        
    Returns:
        Словарь с результатами статистики для каждой ячейки
    """
    return {
        coords: StatisticsResult(
            mean=round(np.mean(tile_img), 2),
            std=round(np.std(tile_img), 2),
            median=round(np.median(tile_img), 2)
        )
        for coords, tile_img in tiles.items()
    }

def calculate_overall_statistics(
    stats_element: Dict[Tuple[int, int], StatisticsResult]
) -> Dict[str, Union[float, List[float], List[Tuple[int, int]]]]:
    """Вычисляет общую статистику по всем ячейкам.
    
    Args:
        stats_element: Словарь со статистикой для каждой ячейки
        
    Returns:
        Словарь с общей статистикой, включая:
        - mean: среднее значение
        - std: стандартное отклонение
        - median: медиана
        - all_mean: список средних значений по ячейкам
        - all_std: список стандартных отклонений по ячейкам
        - all_median: список медиан по ячейкам
        - all_coords: список координат ячеек
    """
    all_mean_values = [stats.mean for stats in stats_element.values()]
    all_std_values = [stats.std for stats in stats_element.values()]
    all_median_values = [stats.median for stats in stats_element.values()]
    all_coords = list(stats_element.keys())

    return {
        'mean': round(np.mean(all_mean_values), 2),
        'std': round(np.std(all_std_values), 2),
        'median': round(np.median(all_median_values), 2),
        'all_mean': all_mean_values,
        'all_std': all_std_values,
        'all_median': all_median_values,
        'all_coords': all_coords
    }

def detect_outliers_in_rows(
    stats_element: Dict[Tuple[int, int], StatisticsResult]
) -> Dict[Tuple[int, int], StatisticsResult]:
    """Находит и нормализует выбросы в каждой строке изображения.
    
    Функция анализирует статистику по строкам и нормализует значения,
    которые значительно отклоняются от среднего.
    
    Args:
        stats_element: Словарь со статистикой для каждой ячейки
        
    Returns:
        Словарь с нормализованными значениями статистики
    """
    grouped_rows: Dict[int, List[Tuple[Tuple[int, int], StatisticsResult]]] = {}
    for coord, values in stats_element.items():
        x_coord = coord[0]
        if x_coord not in grouped_rows:
            grouped_rows[x_coord] = []
        grouped_rows[x_coord].append((coord, values))
        
    def find_and_normalize_outliers(
        row: List[Tuple[Tuple[int, int], StatisticsResult]]
    ) -> None:
        """Находит и нормализует выбросы в строке.
        
        Args:
            row: Список кортежей с координатами и статистикой ячеек
        """
        mean_values = [value.mean for _, value in row]
        std_values = [value.std for _, value in row]
        median_values = [value.median for _, value in row]
        mean_mean = sum(mean_values) / len(mean_values)
        mean_std = sum(std_values) / len(std_values)
        mean_median = sum(median_values) / len(median_values)
        outlier_coords, recession_coords = [], []

        for coord, values in row:
            if (
                values.mean > (mean_mean + mean_std) * 1.5
                or values.std > (mean_std + mean_std) * 1.5
                or values.median > (mean_median + mean_std) * 1.5
            ):
                outlier_coords.append(coord)
            elif (
                values.mean < (mean_mean - mean_std) * 0.3
                or values.std < (mean_std - mean_std) * 0.3
                or values.median < (mean_median - mean_std) * 0.3
            ):
                recession_coords.append(coord)

        for outlier_coord in outlier_coords:
            stats_element[outlier_coord] = StatisticsResult(mean_mean, mean_std, mean_median)
            warning2(outlier_coord)

        for recession_coord in recession_coords:
            stats_element[recession_coord] = StatisticsResult(mean_mean, mean_std, mean_median)
            warning3(recession_coord)

    for row in grouped_rows.values():
        find_and_normalize_outliers(row)

    return stats_element

def grid_visualiser(
    finally_statistics: Dict[str, Union[float, List[float], List[Tuple[int, int]]]],
    bnds: Dict[str, Tuple[float, float]],
    wrns: Dict[str, float]
) -> Dict[str, Union[List[float], List[Tuple[int, int]], List[bool]]]:
    """Визуализирует результаты статистики и проверяет предупреждения.
    
    Args:
        finally_statistics: Общая статистика по ячейкам
        bnds: Границы допустимых значений
        wrns: Пороговые значения для предупреждений
        
    Returns:
        Словарь с визуализированными данными и флагами предупреждений
    """
    all_mean_val: List[float] = []
    all_std_val: List[float] = []
    all_median_val: List[float] = []
    is_warning_cell: List[bool] = []

    for element_all_mean_val, element_all_std_val, element_all_median_val in zip(finally_statistics['all_mean'],
                                                                                 finally_statistics['all_std'],
                                                                                 finally_statistics['all_median']):
        element_all_mean_val = sf_cell(bnds, element_all_mean_val, 'mean')
        element_all_std_val = sf_cell(bnds, element_all_std_val, 'std')
        element_all_median_val = sf_cell(bnds, element_all_median_val, 'median')

        all_mean_val.append(element_all_mean_val)
        all_std_val.append(element_all_std_val)
        all_median_val.append(element_all_median_val)
        is_warning = any((
            element_all_mean_val >= wrns['mean'],
            element_all_std_val >= wrns['std'],
            element_all_median_val >= wrns['median'],
        ))
        is_warning_cell.append(is_warning)
    if any(is_warning_cell):
        warning()
    return {
        'all_mean_val': all_mean_val,
        'all_std_val': all_std_val,
        'all_median_val': all_median_val,
        'all_coords': finally_statistics['all_coords'],
        'is_warning_cell': is_warning_cell,
    }

def sf(x: Dict[str, float], bnds: Dict[str, Tuple[float, float]], name: str) -> float:
    """Нормализует значение в пределах границ.
    
    Args:
        x: Словарь с данными
        bnds: Границы значений
        name: Ключ для значения в словаре
        
    Returns:
        Нормализованное значение в диапазоне [0, 1]
    """
    return (x[name] - bnds[name][0]) / (bnds[name][1] - bnds[name][0])

def sf_cell(bnds: Dict[str, Tuple[float, float]], val: float, name: str) -> float:
    """Нормализует значение в пределах границ.
    
    Args:
        bnds: Границы значений
        val: Значение для нормализации
        name: Ключ для значения в границах
        
    Returns:
        Нормализованное значение в диапазоне [0, 1]
    """
    return (val - bnds[name][0]) / (bnds[name][1] - bnds[name][0])

def warning() -> None:
    """Выводит предупреждение о превышении уровня."""
    print('[WARNING]: Exceeding the level')

def warning2(outlier_coord: Tuple[int, int]) -> None:
    """Выводит предупреждение о выбросе.
    
    Args:
        outlier_coord: Координаты выброса
    """
    print(f"[OUTLIER]: in coordinate: {outlier_coord}")

def warning3(outlier_coord: Tuple[int, int]) -> None:
    """Выводит предупреждение о регрессе.
    
    Args:
        outlier_coord: Координаты регресса
    """
    print(f"[RECESSION]: in coordinate: {outlier_coord}")
    
def check(
    img: np.ndarray,
    cnts: Dict[int, Any],
    cnt_idx: int
) -> None:
    """Выполняет анализ контура на изображении.
    
    Функция выполняет полный анализ контура, включая:
    1. Обрезку и поворот изображения
    2. Разбиение на ячейки
    3. Статистический анализ
    4. Нормализацию выбросов
    5. Визуализацию результатов
    
    Args:
        img: Исходное изображение
        cnts: Словарь с конфигурациями контуров
        cnt_idx: Индекс анализируемого контура
    """
    contour = cnts[cnt_idx].contour
    bnds = cnts[cnt_idx].bounds
    wrns = cnts[cnt_idx].warnings

    rotated = rotate(img, contour)
    grided = grid(rotated, 3, 15)
    statistics_grid = calculate_statistics_in_tiles(grided)
    outliers_control = detect_outliers_in_rows(statistics_grid)
    finally_statistics = calculate_overall_statistics(outliers_control)
    cell_statistics = grid_visualiser(finally_statistics, bnds, wrns)

    mean_val = sf(finally_statistics, bnds, "mean")
    std_val = sf(finally_statistics, bnds, "std")
    median_val = sf(finally_statistics, bnds, "median")

    is_warning = any((
        mean_val >= wrns['mean'],
        std_val >= wrns['std'],
        median_val >= wrns['median'],
    ))
    if is_warning:
        warning()

    cnts[cnt_idx].result = {
        'flags': {
            'is_warning': is_warning,
        },
        'values': {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
        },
        'values_cell': {
            'all_mean': cell_statistics['all_mean_val'],
            'all_std': cell_statistics['all_std_val'],
            'all_median': cell_statistics['all_median_val'],
            'all_coords': cell_statistics['all_coords'],
        },
        'flags_cell': {
            'is_warning_cell': cell_statistics['is_warning_cell'],
        }
    }
