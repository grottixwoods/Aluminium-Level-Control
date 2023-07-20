from processing import *
import cv2



video_path = '123.mp4'

text_pos = (685, 740)
contour = np.array([[685, 645], [980, 610], [980, 680], [710, 710]])


frame_cur = 0
frame_step = 2*3
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    is_next, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cur)
    frame_cur += frame_step
    if not is_next:
        break

    image = frame
    cropped = crop(image, contour)
    rotated = rotate(cropped, contour)
    quantizated = quantization(rotated, clusters=8)
    mean_color = rotated.mean()
    if mean_color >= 0:
        # cv2.imshow('Cropped',cropped)
        # cv2.imshow('Rotated',rotated)
        # cv2.imshow('Quantized',quantizated)
        cv2.putText(
            image,
            f's: {frame_cur//30} | mean_color: {round(mean_color, 2)}',
            text_pos, cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 1)
        cv2.imshow('Image', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()