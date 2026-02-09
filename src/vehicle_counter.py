#verificación de contenido
import os
import json
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm

VIDEO_PATH = r"C:\Users\cisg1\Desktop\tesis\videos\vertiente_1\sem1_dia1\amorning_000.mp4"
assert os.path.exists(VIDEO_PATH), f"Error, no se encuentra el archivo."

print("El archivo fue encontrado:", os.path.abspath(VIDEO_PATH))

#---------------------------------------------------------------------------------------------------
# configuración del conteo
# clases a considerar para clasificar vehículos
CLASSES_ENG = {"car", "bus", "truck", "motorcycle"}
CLASSES_ESP = {"car":"carro", "truck":"camion", "bus":"bus", "motorcycle":"moto"}

# modelo de yolo, si veo que es muy lento usaré "yolov8n.pt"
MODEL_NAME = "best.pt"

# umbrales de detección
CONF_THRESH = 0.20        # confianza mínima para aceptar una detección
IOU_THRESH = 0.45         # NMS: qué tan solapadas pueden estar dos cajas (supresión no máxima)

# ROI provicional: polígono para ignorar zonas (aceras, cielo, sobras, etc)
ROI_POLYGON = None         # ejemplo: [(100, 300), (1180, 300), (1180, 720), (100, 720)]

# distancia real entre líneas en metros (es para calcular velocidad)
DIST_BETWEEN_LINES_M = None           # sería 2.0 o algo así pero no la tengo

#----------------------------------------------------------------------------------------------
#utilidades: líneas, cruce y dibujo
def side_of_line(a, b, p):
  """
  Devuelve el signo del producto cruzado (AB x AP).
  >0: p a la izquierda de la línea AB
  <0: p a la derecha
  0: p sobre la línea
  """
  return np.cross(np.array(b)-np.array(a), np.array(p)-np.array(a))

def crossed(a, b, prev_pt, now_pt, tol=0.0):
  """
  True si el centroide pasó de un lado a otro de la línea AB (cambio de signo).
  tol: si el valor está muy cerca de 0 (sobre la línea), lo ignoramos.
  """
  s1 = side_of_line(a, b, prev_pt)
  s2 = side_of_line(a, b, now_pt)
  if abs(s1) <= tol or abs(s2) <= tol:
    return False
  return (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0)

#--------------------------------------------------------------------------------------------------
# preparar entrada y salida de video y objetos principales
# Leer metadatos del video y crear un escritor para el video anotado
cap_probe = cv2.VideoCapture(VIDEO_PATH)
assert cap_probe.isOpened(), f"No pude abrir {VIDEO_PATH}"
fps = cap_probe.get(cv2.CAP_PROP_FPS)
W = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
cap_probe.release()

VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

ANNOTATED_DIR = r"C:\Users\cisg1\Desktop\tesis\outputs\annotated"
annot_path = os.path.join(
    ANNOTATED_DIR,
    f"{VIDEO_NAME}_annotated.mp4"
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(annot_path, fourcc, fps, (W, H))

# cargar modelo YOLO
model = YOLO(MODEL_NAME)

# inicializar tracker ByteTrack de supervision
tracker = sv.ByteTrack()

# estructuras de estado
last_center = {}                          # track_id -> centroide previo (x,y)
state = {}                                # track_id -> {tipo_en, tipo_es, t1, t2, crossed1, crossed2}
counts = {k: 0 for k in CLASSES_ENG}      # conteo por clase (al cruzar L1)
rows   = []                               # filas para CSV (eventos L2 con delta/velocidad)

#------------------------------------------------------------------------------------------------------------
#bucle principal: detectar, trackear, contar y medir tiempos
# ---- Parámetros simplificados (solo líneas) ----
ENTRY_X_FRAC = 0.23   # Línea L1 (entrada) - verde
EXIT_X_FRAC  = 0.80   # Línea L2 (conteo) - roja

def build_simple_lines(frame_shape):
    """Construye solo las dos líneas de tracking"""
    h, w = frame_shape[:2]

    # Coordenadas X de las líneas
    x_entry = int(ENTRY_X_FRAC * w)
    x_exit = int(EXIT_X_FRAC * w)

    # Definir las líneas
    L1 = ((x_entry, 500), (x_entry, 1080))  # L1: verde (entrada)
    L2 = ((x_exit, 500), (x_exit, 1080))    # L2: roja (conteo)

    return L1, L2

# ================== INICIALIZACIÓN ==================
cap = cv2.VideoCapture(VIDEO_PATH)

# Leer primer frame para obtener dimensiones
ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("No se pudo leer el video")

# Construir las líneas basadas en las dimensiones del frame
L1, L2 = build_simple_lines(first_frame.shape)

roi_polygon = np.array([
    [L1[0][0], L1[0][1]],   # esquina arriba-izquierda (L1)
    [L2[0][0], L2[0][1]],   # esquina arriba-derecha (L2)
    [L2[1][0], L2[1][1]],   # esquina abajo-derecha (L2)
    [L1[1][0], L1[1][1]]    # esquina abajo-izquierda (L1)
], dtype=np.int32)
roi_mask = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(roi_mask, [roi_polygon], 255)

# Volver al inicio del video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Debug info
print(f"=== LÍNEAS DE TRACKING ===")
print(f"Frame: {first_frame.shape[1]}x{first_frame.shape[0]}")
print(f"L1 (Verde): {L1} - Altura: {L1[1][1] - L1[0][1]} px")
print(f"L2 (Roja): {L2} - Altura: {L2[1][1] - L2[0][1]} px")

pbar = tqdm(total=total_frames or None, desc="Procesando")
frame_idx = 0

# anotadores visuales (cajas y etiquetas)
box_annotator = sv.BoxAnnotator()
label_anotator = sv.LabelAnnotator()

# ================== LOOP PRINCIPAL ==================
while True:
  ok, frame = cap.read()
  if not ok:
    break
  frame_idx += 1

  # tiempo del frame en segundos derivado del índice y fps
  timestamp = frame_idx / fps

  # inferencia YOLO en el frame completo (SIN ROI)
  result = model.predict(source=frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

  # convierto a objeto Detections de supervision
  det = sv.Detections.from_ultralytics(result)

  # filtrar por clases de interés las que definí en el inicio
  if len(det) > 0:
    cls_mask = np.array([model.names[int(c)] in CLASSES_ENG for c in det.class_id])
    det = det[cls_mask]

  # actualizar tracker con las detecciones filtradas
  det = tracker.update_with_detections(det)

  # coordenadas de centros (para los cruces)
  centers = det.get_anchors_coordinates(sv.Position.CENTER)

  # DIBUJAR LÍNEAS DE CONTEO CON MEJOR VISUALIZACIÓN
  # L1 (Verde) - más alta
  cv2.line(frame, L1[0], L1[1], color=(0,255,0), thickness=4)
  cv2.putText(frame, "L1: ENTRADA", (L1[0][0] + 10, L1[0][1] - 15),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

  # L2 (Roja) - altura normal
  cv2.line(frame, L2[0], L2[1], color=(0,0,255), thickness=4)
  cv2.putText(frame, "L2: CONTEO", (L2[0][0] + 10, L2[0][1] - 15),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

  # >>> DEBUG VISUAL DEL ROI (azul semitransparente) <<<
  overlay = frame.copy()
  cv2.fillPoly(overlay, [roi_polygon], (255, 0, 0))   # azul sólido
  alpha = 0.2  # transparencia (0=transparente, 1=opaco)
  frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

  labels = []
  for i in range(len(det)):
    track_id = det.tracker_id[i]
    if track_id is None:
      labels.append("")  # detección sin ID estable, no usamos
      continue

    cls_id = int(det.class_id[i])
    cls_en = model.names[cls_id]
    if cls_en not in CLASSES_ENG:
      labels.append("")
      continue

    # centroide actual
    cx, cy = map(int, centers[i])
    now_pt = (cx, cy)

    # inicializar estado si es la primera vez que vemos este ID
    if track_id not in state:
      state[track_id] = {
          "tipo_en": cls_en,
          "tipo_es": CLASSES_ESP.get(cls_en, cls_en),
          "t1": None, "t2": None,
          "crossed1": False, "crossed2": False,
          "frames_seen": 0
      }

    # incrementar frames vistos
    state[track_id]["frames_seen"] += 1

    min_frames = 3 if cls_en == "motorcycle" else 5

    if state[track_id]["frames_seen"] < min_frames:
      last_center[track_id] = now_pt
      labels.append("")  # ← para supervision así no manda error
      continue

    # si ya teníamos un centro previo, evaluamos cruces
    if track_id in last_center:
      prev_pt = last_center[track_id]

      # cruce de LÍNEA 1 (solo una vez)
      if not state[track_id]["crossed1"] and crossed(np.array(L1[0]), np.array(L1[1]), prev_pt, now_pt):
        state[track_id]["crossed1"] = True
        state[track_id]["t1"] = timestamp

      # cruce de LÍNEA 2 (solo una vez, con validación espacial)
      if state[track_id]["crossed1"] and not state[track_id]["crossed2"]:
        
        if crossed(np.array(L2[0]), np.array(L2[1]), prev_pt, now_pt):
          state[track_id]["crossed2"] = True
          state[track_id]["t2"] = timestamp

          delta = state[track_id]["t2"] - state[track_id]["t1"]

          speed_kmh = None
          if (DIST_BETWEEN_LINES_M is not None) and delta > 0:
            speed_ms = DIST_BETWEEN_LINES_M / delta
            speed_kmh = 3.6 * speed_ms
          
          counts[cls_en] += 1

          rows.append([
            int(track_id),
            state[track_id]["tipo_en"],
            state[track_id]["tipo_es"],
            round(state[track_id]["t1"], 3) if state[track_id]["t1"] else None,
            round(state[track_id]["t2"], 3) if state[track_id]["t2"] else None,
            round(delta, 3) if delta else None,
            round(speed_kmh, 2) if speed_kmh else None
          ])

    # se actualiza el centro previo de este ID
    last_center[track_id] = now_pt

    # etiqueta visual para el video
    label_txt = f"{state[track_id]['tipo_es']} #{track_id}"
    if state[track_id]["t1"] and not state[track_id]["t2"]:
      label_txt += " | entre L1-L2"
    labels.append(label_txt)

  # se dibujan las cajas y etiquetas sobre el frame
  if len(det) > 0:
    frame = box_annotator.annotate(scene=frame, detections=det)
    frame = label_anotator.annotate(scene=frame, detections=det, labels=labels)

  # cabecera con conteos por clase en vivo
  y0 = 28
  for k in CLASSES_ENG:
    txt = f"{CLASSES_ESP.get(k,k)}: {counts[k]}"
    cv2.putText(frame, txt, (18, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,230,50), 2)
    y0 += 30

  # escribir frame anotado al archivo de salida
  writer.write(frame)
  pbar.update(1)

cap.release()
writer.release()
pbar.close()
print("Video anotado guardado en: ", annot_path)

#----------------------------------------------------------------------------------------------------
#guardar resultados, csv con eventos y resumen json
# ------------------ CSV GLOBAL DE EVENTOS ------------------
CSV_PATH = r"C:\Users\cisg1\Desktop\tesis\outputs\csv\eventos.csv"
csv_exists = os.path.isfile(CSV_PATH)

with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
    writer_csv = csv.writer(f)

    # encabezado solo la primera vez
    if not csv_exists:
        writer_csv.writerow([
            "video",
            "track_id",
            "clase_en",
            "clase_es",
            "t1_s",
            "t2_s",
            "delta_s",
            "velocidad_kmh"
        ])

    # escribir eventos
    for r in rows:
        writer_csv.writerow([
            os.path.basename(VIDEO_PATH),
            *r
        ])

# ------------------ JSON GLOBAL DE RESÚMENES ------------------
JSON_PATH = r"C:\Users\cisg1\Desktop\tesis\outputs\json\resumen.json"

# cargar existente si existe
if os.path.isfile(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        try:
            summaries = json.load(f)
        except json.JSONDecodeError:
            summaries = []
else:
    summaries = []

# eliminar resumen previo del mismo video (si existe)
summaries = [
    s for s in summaries
    if s.get("video") != os.path.basename(VIDEO_PATH)
]

# nuevo resumen
summary = {
    "video": os.path.basename(VIDEO_PATH),
    "fps": fps,
    "width": W,
    "height": H,
    "line1": L1,
    "line2": L2,
    "counts": counts,
    "total": int(sum(counts.values()))
}

summaries.append(summary)

# guardar
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)


print("Video anotado guardado en:", annot_path)
print("Eventos agregados a:", CSV_PATH)
print("Resumen actualizado en:", JSON_PATH)
print("Conteos finales:", counts)