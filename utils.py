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

def overall_statistics(stats_element): # общая для cv.putText()
    all_mean_values = []
    all_std_values = []
    all_median_values = []
    for coords, (mean_value, std_value, median_value) in stats_element.items():
        all_mean_values.append(mean_value)
        all_std_values.append(std_value)
        all_median_values.append(median_value)

    overall_mean = round(np.mean(all_mean_values), 2)
    overall_std = round(np.std(all_std_values), 2)
    overall_median = round(np.median(all_median_values), 2)
    return {
        'mean':overall_mean,
        'std':overall_std,
        'median':overall_median
    }

def detect_outliers_in_rows(stats_element):
    row_warnings = {}
    sorted_items = sorted(stats_element.items(), key=lambda item: item[0][0])
    grouped_rows = {}
    for coords, stats in sorted_items:
        row_idx = coords[0]
        if row_idx not in grouped_rows:
            grouped_rows[row_idx] = []
        grouped_rows[row_idx].append((coords, stats))
    for row_idx, row_elements in grouped_rows.items():
        median_values = [stats[2] for _, stats in row_elements]
        median_median = np.median(median_values)
        threshold = 2 * median_median
        for (coords, stats) in row_elements:
            median_val = stats[2]
            if abs(median_val - median_median) > threshold:
                if row_idx not in row_warnings:
                    row_warnings[row_idx] = []
                row_warnings[row_idx].append(coords)
    return row_warnings

sf = lambda x, bnds, name: (x[name] - bnds[name][0]) / (bnds[name][1] - bnds[name][0])


def warning():
    print(f'[WARNING]: Exceeding the level')

def check(img, cnts, cnt_idx):
    cnt = cnts[cnt_idx]['contour']
    bnds = cnts[cnt_idx]['bounds']
    wrns = cnts[cnt_idx]['warnings']

    rotated = rotate(img, cnt)
    grided = grid(rotated, 20)
    statistics_grid = statistics_in_tiles(grided)
    overall_val_grid = overall_statistics(statistics_grid)

    mean_val = sf(overall_val_grid, bnds, "mean")
    std_val = sf(overall_val_grid, bnds, "std")
    median_val = sf(overall_val_grid, bnds, "median")

    is_outlier = False
    is_warning = any((
        mean_val >= wrns['mean'],
        std_val >= wrns['std'],
        median_val >= wrns['median'],
    ))
    if is_warning:
        warning()


    detected_outliers = detect_outliers_in_rows(statistics_grid)
    for row_idx, outlier_coords in detected_outliers.items():
        print(f"[OUTLIER]: Row {row_idx} at coordinates {outlier_coords}")

    cnts[cnt_idx]['result'] = {
        'flags': {
            'is_warning': is_warning,
            'is_outlier': is_outlier,
        },
        'values': {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
        }
    }
