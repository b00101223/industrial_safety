from ultralytics import YOLO
import cv2

model = YOLO("hf://Hexmon/vyra-yolo-ppe-detection/best.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes

    print(f"--- {len(boxes)} objects ---")
    for b in boxes:
        cls_name = model.names[int(b.cls)]
        conf = float(b.conf)
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        print(f"  {cls_name:15s} conf={conf:.2f}  box=({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")

    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()