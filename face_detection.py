import cv2
import mediapipe as mp

mpFaces = mp.solutions.face_detection
faces = mpFaces.FaceDetection()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faces.process(frame)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(frame, detection)
            # print(detection.location_data.relative_bounding_box)
            boxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            box = int(boxC.xmin * w), int(boxC.ymin * h), int(boxC.width * w), int(boxC.height * h)
            cv2.rectangle(frame, box, (182, 24, 82), 1)
            cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                        (12, 112, 88), 2)
            x, y, w1, h1 = box
            x1, y1 = x + w1, y + h1
            cv2.line(frame, (x, y), (x + 30, y), (182, 24, 82), 4)
            cv2.line(frame, (x, y), (x, y + 30), (182, 24, 82), 4)
            cv2.line(frame, (x1, y), (x1 - 30, y), (182, 24, 82), 4)
            cv2.line(frame, (x1, y), (x1, y + 30), (182, 24, 82), 4)
            cv2.line(frame, (x, y1), (x + 30, y1), (182, 24, 82), 4)
            cv2.line(frame, (x, y1), (x, y1 - 30), (182, 24, 82), 4)
            cv2.line(frame, (x1, y1), (x1 - 30, y1), (182, 24, 82), 4)
            cv2.line(frame, (x1, y1), (x1, y1 - 30), (182, 24, 82), 4)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xff is 27:
        break

cap.release()
cv2.destroyAllWindows()