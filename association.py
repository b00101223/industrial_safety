from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

model_path = hf_hub_download(
    repo_id="Hexmon/vyra-yolo-ppe-detection",
    filename="best.pt"
)
model = YOLO(model_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)[0]

    for b in r.boxes:
        cls = model.names[int(b.cls)]
        conf = float(b.conf)
        tid = int(b.id) if b.id is not None else None
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

        print(f"ID {tid}  {cls}  conf={conf:.2f}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Layer 1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()