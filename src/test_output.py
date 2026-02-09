import cv2
import os

VIDEO_PATH = r"C:\Users\lasso\Desktop\tesis\videos\vertiente_1\sem1_dia1\amorning_000.mp4"
OUTPUT_PATH = r"C:\Users\lasso\Desktop\tesis\test_output.avi"

assert os.path.exists(VIDEO_PATH), "No se encuentra el video"

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "No se pudo abrir el video"

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}, Width: {width}, Height: {height}")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

cap.release()
out.release()

print("Video de prueba generado correctamente")
