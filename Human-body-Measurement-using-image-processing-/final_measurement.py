import cv2
import mediapipe as mp
import time
from measure_calcuation import chest, suit_length, ration_factor, edge_detection, find_radius, check_y_coordinate, find_mid_point, find_distance

def centroid(point1, point2, point3, point4):
    x, y = (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2))
    x1, y1 = (int((point3[0] + point4[0]) / 2), int((point3[1] + point4[1]) / 2))
    a, b = int((x + x1) / 2), int((y + y1) / 2)
    return a, b

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
img = cv2.imread('pranav.jpeg')
height = 163
sizes = img.shape

if sizes[1] > 1500:
    w = 230
    h = 620
else:
    w = 330
    h = 716
dim = (w, h)
frame = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(imgRGB)
points = []

for id, lm in enumerate(results.pose_landmarks.landmark):
    h, w, c = frame.shape
    cx, cy = int(lm.x * w), int(lm.y * h)
    points.append([cx, cy])

(a, b) = centroid(points[12], points[11], points[24], points[23])

if results.pose_landmarks:
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

ratio = ration_factor(height, points[0], points[28])

# Calculate body measurements
lengths = suit_length(ratio, points[12], points[24], 0)
arms = suit_length(ratio, points[12], points[16], 0)
chests = chest(ratio, points[12], points[11], 0)
shoulder = suit_length(ratio, points[12], points[11], 0)
paint_length = suit_length(ratio, points[24], points[28], 0)

# Calculate circular measurements
p = (a, b)
contours, edge, edgess = edge_detection(img)
sides, _ = check_y_coordinate(contours, p)
circumference = find_radius(p, sides, ratio)
chest_mid = find_mid_point(points[11], points[12])
temp = (chest_mid[0], chest_mid[1])
temp2 = (points[11][0], points[11][1])
ch = suit_length(ratio, temp2, temp, 0)
chests = ch * 4
cht = 2 * ch * 3.14
heap = suit_length(ratio, points[24], points[23], 0)
heap = heap * 2 * 3.14
temp = find_mid_point(points[23], points[25])
hside, _ = check_y_coordinate(contours, temp)
thigh = suit_length(ratio, temp, hside, 0)
thigh = 2 * 3.14 * thigh
cv2.circle(frame, temp, 2, (255, 0, 0), 3)
cv2.circle(frame, hside, 2, (255, 0, 0), 3)
cv2.line(frame, temp, hside, (255, 0, 0), 3)

# Define a function to draw text on the image with custom formatting
def draw_formatted_text(image, text, position, font_scale=0.7, font_color=(255, 255, 255), font_thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)

# Display the values of the body parts on the image with custom formatting
draw_formatted_text(frame, f'Suit Length: {round(lengths,2)}', (10, 30), font_scale=0.3, font_color=(20,0,0), font_thickness=1)
draw_formatted_text(frame, f'Arm Length: {round(arms,2)}', (10, 60), font_scale=0.3, font_color=(0,0,0), font_thickness=1)
draw_formatted_text(frame, f'Chest: {round(chests,2)}', (10, 90), font_scale=0.3, font_color=(0,0,0), font_thickness=1)
draw_formatted_text(frame, f'Shoulder Length: {round(shoulder,2)}', (10, 120), font_scale=0.3, font_color=(0,0,0), font_thickness=1)
draw_formatted_text(frame, f'Pant Length: {round(paint_length,2)}', (10, 150), font_scale=0.3, font_color=(0,0,0), font_thickness=1)

# Display the values of the circular measurements on the image with custom formatting
draw_formatted_text(frame, f'Upper Body Waist: {round(circumference,2)}', (10, 180), font_scale=0.3, font_color=(0,0,0), font_thickness=1)
draw_formatted_text(frame, f'Chest Circumference: {round(cht,2)}', (10, 210), font_scale=0.3, font_color=(0,0,0), font_thickness=1)
draw_formatted_text(frame, f'Heap Circumference: {round(heap,2)}', (10, 240), font_scale=0.3, font_color=(0,0,0), font_thickness=1)
draw_formatted_text(frame, f'Thigh Circumference: {round(thigh,2)}', (10, 270), font_scale=0.3, font_color=(0,0,0), font_thickness=1)

cv2.imshow('Human Body Measurement', frame)
cv2.waitKey(0)
