import cv2
import mediapipe as mp
import numpy as np

mp_selfie = mp.solutions.selfie_segmentation
model = mp_selfie.SelfieSegmentation()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = model.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.5
    segmented_image = np.where(mask, frame, cv2.blur(frame, (40, 40)))

    cv2.imshow("blur", segmented_image)

    if cv2.waitKey(10) & 0xFF is 27:
        break
cap.release()
cv2.destroyAllWindows()