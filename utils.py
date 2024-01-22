import cv2
import numpy as np

# Функция crop(img, pts): обрезает изображение по указанным координатам pts.
# img - изображение, pts - координаты для обрезки.
# Возвращает обрезанное изображение.

def crop(img, pts): 
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(pts)
    cropped = res[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    return cropped

# Функция order_points(pts): упорядочивает координаты pts так, чтобы они следовали в порядке "влево-вверх, вправо-вниз".
# pts - массив с координатами четырехугольника.
# Возвращает упорядоченный массив координат.

def order_points(pts): 
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Функция rotate(img, pts): перспективное преобразование изображения в вид прямоугольника, ограниченного pts.
# img - изображение, pts - координаты четырехугольника.
# Возвращает изображение с правильной перспективой.

def rotate(img, pts):
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


# Функция calibrate_frame(frame, target_mean, target_std): калибрует изображение, изменяя яркость и контраст.
# (функция применима к потоковому видео с камеры, для которой требуется калибровка.)
# frame - изображение, target_mean - целевое среднее значение яркости, target_std - целевое стандартное отклонение яркости.
# Возвращает калиброванное изображение. 

def calibrate_frame(frame, target_mean, target_std):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray_frame)
    std_val = np.std(gray_frame)
    contrast = target_std / std_val
    brightness = target_mean - (mean_val * contrast)
    adjusted_frame = cv2.convertScaleAbs(gray_frame, alpha=contrast, beta=brightness)
    calibrated_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)
    return calibrated_frame

# Функция quantization(img, clusters=8, rounds=1): применяет квантование цветов изображения с использованием k-средних.
# (Эксперементальная функция, необходимость применения стоит рассматривать после получения потокового видео с камеры)
# img - изображение, clusters - количество кластеров для квантования, rounds - количество итераций k-средних.
# Возвращает изображение после квантования.

def quantization(img, clusters=8, rounds=1):
    samples = img.reshape(-1, 3).astype(np.float32)
    compactness, labels, centers = cv2.kmeans(
        samples, clusters, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
        rounds, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(img.shape)

# Функция grid(img, cells_in_height, cells_in_width): разбивает изображение на ячейки заданного размера.
# img - изображение, cells_in_height - количество ячеек в высоте, cells_in_width - количество ячеек в ширине.
# Возвращает словарь, где ключи - координаты ячеек, значения - изображения ячеек.

def grid(img, cells_in_height, cells_in_width):
    h, w = img.shape[:2]
    d_h = h // cells_in_height
    d_w = w // cells_in_width
    tiles = {}
    for i in range(0, h, d_h):
        for j in range(0, w, d_w):
            box = (i, j, i + d_h, j + d_w)
            tile_img = img[box[0]:box[2], box[1]:box[3]]
            tiles[(i, j)] = tile_img
    return tiles

# Функция calculate_statistics_in_tiles(tiles): вычисляет статистику (среднее, стандартное отклонение, медиану) для каждой ячейки.
# tiles - словарь с координатами и изображениями ячеек.
# Возвращает словарь с результатами статистики для каждой ячейки.

def calculate_statistics_in_tiles(tiles):
    stats_element = {}
    for coords, tile_img in tiles.items():
        mean_value = round(np.mean(tile_img), 2)
        std_value = round(np.std(tile_img), 2)
        median_value = round(np.median(tile_img), 2)
        stats_element[coords] = mean_value, std_value, median_value
    return stats_element

# Функция calculate_overall_statistics(stats_element): вычисляет общую статистику по всем значениям из словаря статистики.
# stats_element - словарь со статистикой для каждой ячейки.
# Возвращает словарь с общей статистикой.

def calculate_overall_statistics(stats_element):
    all_mean_values, all_std_values, all_median_values, all_coords = [], [], [], []
    for coords, (mean_value, std_value, median_value) in stats_element.items():
        all_mean_values.append(mean_value)
        all_std_values.append(std_value)
        all_median_values.append(median_value)
        all_coords.append(coords)

    overall_mean = round(np.mean(all_mean_values), 2)
    overall_std = round(np.std(all_std_values), 2)
    overall_median = round(np.median(all_median_values), 2)

    return {
        'mean': overall_mean,
        'std': overall_std,
        'median': overall_median,
        'all_mean': all_mean_values,
        'all_std': all_std_values,
        'all_median': all_median_values,
        'all_coords': all_coords
    }

# Функция detect_outliers_in_rows(stats_element): находит и нормализует выбросы в каждой строке изображения.
# stats_element - словарь со статистикой для каждой ячейки.
# Возвращает словарь с результатами после нормализации выбросов.

def detect_outliers_in_rows(stats_element):
    grouped_rows = {}
    for coord, values in stats_element.items():
        x_coord = coord[0]
        if x_coord not in grouped_rows:
            grouped_rows[x_coord] = []
        grouped_rows[x_coord].append((coord, values))
        
    # Функция find_and_normalize_outliers(row): находит и нормализует выбросы в каждой строке изображения.
    # row - список кортежей с координатами и значениями статистики для каждой ячейки в строке.
    # Нормализует значения и обновляет статистику в словаре stats_element. Выводит предупреждения о выбросах и регрессах.

    def find_and_normalize_outliers(row):
        mean_values = [value[0] for _, value in row]
        std_values = [value[1] for _, value in row]
        median_values = [value[2] for _, value in row]
        mean_mean = sum(mean_values) / len(mean_values)
        mean_std = sum(std_values) / len(std_values)
        mean_median = sum(median_values) / len(median_values)
        outlier_coords, recession_coords = [], []

        for coord, values in row:
            if (
                values[0] > (mean_mean + mean_std) * 1.5
                or values[1] > (mean_std + mean_std) * 1.5
                or values[2] > (mean_median + mean_std) * 1.5
            ):
                outlier_coords.append(coord)
            elif (
                values[0] < (mean_mean - mean_std) * 0.3
                or values[1] < (mean_std - mean_std) * 0.3
                or values[2] < (mean_median - mean_std) * 0.3
            ):
                recession_coords.append(coord)

        for outlier_coord in outlier_coords:
            outlier_mean, outlier_std, outlier_median = stats_element[outlier_coord]
            stats_element[outlier_coord] = (mean_mean, mean_std, mean_median)
            warning2(outlier_coord)

        for recession_coord in recession_coords:
            outlier_mean, outlier_std, outlier_median = stats_element[recession_coord]
            stats_element[recession_coord] = (mean_mean, mean_std, mean_median)
            warning3(recession_coord)

    for row in grouped_rows.values():
        find_and_normalize_outliers(row)

    return stats_element

# Функция grid_visualiser(finally_statistics, bnds, wrns): визуализирует результаты статистики и проверяет предупреждения.
# finally_statistics - общая статистика, bnds - границы значений, wrns - уровни предупреждений.
# Возвращает словарь с визуализированными данными.

def grid_visualiser(finally_statistics, bnds, wrns):
    all_mean_val, all_std_val, all_median_val, all_coords, is_warning_cell = [], [], [], [], []

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
        'all_coords': all_coords,
        'is_warning_cell': is_warning_cell,
    }

# Функция sf(x, bnds, name): нормализует значение x[name] в пределах границ bnds[name].
# x - словарь с данными, bnds - границы значений, name - ключ для значения в словаре.
# Возвращает нормализованное значение.

def sf(x, bnds, name):
    return (x[name] - bnds[name][0]) / (bnds[name][1] - bnds[name][0])

# Функция sf_cell(bnds, val, name): нормализует значение val в пределах границ bnds[name].
# bnds - границы значений, val - значение, name - ключ для значения в границах.
# Возвращает нормализованное значение.

def sf_cell(bnds, val, name):
    return (val - bnds[name][0]) / (bnds[name][1] - bnds[name][0])

# Функция warning(): выводит предупреждение о превышении уровня.

def warning():
    print(f'[WARNING]: Exceeding the level')

# Функция warning2(outlier_coord): выводит предупреждение о выбросе в указанных координатах.

def warning2(outlier_coord):
    print(f"[OUTLIER]:  in coordinate: {outlier_coord}")

# Функция warning3(outlier_coord): выводит предупреждение о регрессе в указанных координатах.

def warning3(outlier_coord):
    print(f"[RECESSION]: in coordinate: {outlier_coord}")

# Функция check(img, cnts, cnt_idx): выполняет анализ контура на изображении с использованием предоставленных параметров.
# img - изображение, cnts - список контуров с параметрами, cnt_idx - индекс контура для анализа.
# Результат анализа записывается в структуру контура в виде флагов и значений.

def check(img, cnts, cnt_idx):
    cnt = cnts[cnt_idx]['contour']
    bnds = cnts[cnt_idx]['bounds']
    wrns = cnts[cnt_idx]['warnings']

    rotated = rotate(img, cnt)
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

    cnts[cnt_idx]['result'] = {
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
