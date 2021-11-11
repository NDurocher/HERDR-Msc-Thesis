import torch
import cv2

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]
# Images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
video = cv2.VideoCapture(0)
key = cv2.waitKey(10)
# Inference
while True:
    check, frame = video.read()
    results = model(frame)

    # Results
    results.print()
    # results.show()  # or .show()
    # print(results)
    # results.xyxy[0]  # img1 predictions (tensor)
    # results.pandas().xyxy[0]
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()