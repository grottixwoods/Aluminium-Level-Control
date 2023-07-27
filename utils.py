import cv2
import numpy as np


def crop(img, pts):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(img, img, mask = mask)
    rect = cv2.boundingRect(pts)
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

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
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	return warped
    
def calibrate_frame(frame, target_mean, target_std):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(frame)
    std_val = np.std(frame)
    contrast = target_std / std_val
    brightness = target_mean - (mean_val * contrast)
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    calibrated_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)
    return calibrated_frame

def quantization(img, clusters=8, rounds=1):
    h, w = img.shape[:2]
    samples = np.zeros([h*w, 3], dtype=np.float32)
    count = 0
    for x in range(h):
        for y in range(w):
            samples[count] = img[x][y]
            count += 1
    compactness, labels, centers = cv2.kmeans(
        samples,
        clusters,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
        rounds,
        cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((img.shape))


def grid(img, d):
    h, w = img.shape[:2]
    grid = np.array(np.meshgrid(range(0, h-h%d, d), range(0, w-w%d, d))).T.reshape(-1, 2)
    tiles = {}
    for i, j in grid:
        box = (i, j, i+d, j+d)
        tile_img = img[box[0]:box[2], box[1]:box[3]]
        tiles[(i, j)] = tile_img
    return tiles

def statistics_in_tiles(tiles): #для каждого элемента в сетке
    stats_element = {}
    for coords, tile_img in tiles.items():
        mean_value = round(np.mean(tile_img),2)
        std_value = round(np.std(tile_img),2)
        median_value = round(np.median(tile_img),2)
        stats_element[coords] = mean_value, std_value, median_value
    return stats_element

def overall_statistics(stats_element):  # общая для cv.putText()
    all_mean_values = []
    all_std_values = []
    all_median_values = []
    all_coords = []
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

def detect_outliers_in_rows(stats_element):
    grouped_rows = {}
    for coord, values in stats_element.items():
        x_coord = coord[0]
        if x_coord not in grouped_rows:
            grouped_rows[x_coord] = []
        grouped_rows[x_coord].append((coord, values))

    def find_and_normalize_outliers(row):
        mean_values = [value[0] for _, value in row]
        std_values = [value[1] for _, value in row]
        median_values = [value[2] for _, value in row]
        mean_mean = sum(mean_values) / len(mean_values)
        mean_std = sum(std_values) / len(std_values)
        mean_median = sum(median_values) / len(median_values)
        outlier_coord = None
        recession_coord = None
        for coord, values in row:
            if (
                values[0] > (mean_mean + mean_std) * 1.5
                or values[1] > (mean_std + mean_std) * 1.5
                or values[2] > (mean_median + mean_std) * 1.5
            ):
                outlier_coord = coord
                break
            elif (
                values[0] < (mean_mean - mean_std) * 0.3
                or values[1] < (mean_std - mean_std) * 0.3
                or values[2] < (mean_median - mean_std) * 0.3                   
            ):
                recession_coord = coord
                break                
                

        if outlier_coord:
            outlier_mean, outlier_std, outlier_median = stats_element[outlier_coord]
            stats_element[outlier_coord] = (
                mean_mean,
                mean_std,
                mean_median,
            )
            warning2(outlier_coord)
        
        elif recession_coord:
            outlier_mean, outlier_std, outlier_median = stats_element[recession_coord]
            stats_element[recession_coord] = (
                mean_mean,
                mean_std,
                mean_median,
            )
            warning3(recession_coord)        

    for row in grouped_rows.values():
        find_and_normalize_outliers(row)

    return stats_element

def grid_visualiser(finally_statistics, bnds, wrns):
    all_mean_val = []
    all_std_val = []
    all_median_val = []
    all_coords = finally_statistics['all_coords']
    is_warning_cell = []

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
    if is_warning:
        warning()
    return {
        'all_mean_val': all_mean_val,
        'all_std_val': all_std_val,
        'all_median_val': all_median_val,
        'all_coords': all_coords,
        'is_warning_cell': is_warning_cell,
    }


sf = lambda x, bnds, name: (x[name] - bnds[name][0]) / (bnds[name][1] - bnds[name][0])
sf_cell = lambda bnds, val, name: (val - bnds[name][0]) / (bnds[name][1] - bnds[name][0])

def warning():
    print(f'[WARNING]: Exceeding the level')

def warning2(outlier_coord):
    print(f"[OUTLIER]:  in coordinate: {outlier_coord}")

def warning3(outlier_coord):
    print(f"[RECESSION]: in coordinate: {outlier_coord}")


def check(img, cnts, cnt_idx):
    cnt = cnts[cnt_idx]['contour']
    bnds = cnts[cnt_idx]['bounds']
    wrns = cnts[cnt_idx]['warnings']

    rotated = rotate(img, cnt)
    calibrated = calibrate_frame(rotated, 128, 50)
    grided = grid(calibrated, 20)
    statistics_grid = statistics_in_tiles(grided)
    outliers_control = detect_outliers_in_rows(statistics_grid)
    finally_statistics = overall_statistics(outliers_control)
    cell_statistics = grid_visualiser(finally_statistics, bnds, wrns)


    mean_val = sf(finally_statistics, bnds, "mean")
    std_val = sf(finally_statistics, bnds, "std")
    median_val = sf(finally_statistics, bnds, "median")
    
    is_outlier = False
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
            'is_outlier': is_outlier,
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
