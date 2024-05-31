import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu'nun eşikleme yöntemi
    _, otsu_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold = float(np.mean(otsu_threshold))  # Otsu eşiğini skaler hale getirme
    low_threshold = 0.5 * otsu_threshold
    high_threshold = otsu_threshold
    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 60), 10)
    return line_image

def region_of_interest(image):
    polygons = np.array([[(1015, 560), (1175, 560), (1682, 870), (478, 870)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def filter_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Sarı ve beyaz renk aralıkları
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 255, 255])
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    # Maskeler oluşturma
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    return filtered_image

cap = cv2.VideoCapture("seritvideo.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break
    filtered_image = filter_colors(frame)
    canny_image = canny(filtered_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=5)

    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
