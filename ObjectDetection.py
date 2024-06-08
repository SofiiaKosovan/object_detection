import cv2
from ultralytics import YOLO

yolo = YOLO('yolov8n.pt')

video_path = '/Users/skosovan/Documents/_video/video11.mp4'
videoCap = cv2.VideoCapture(video_path)


def get_colours(num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


while True:
    success, frame = videoCap.read()

    if not success:
        continue

    outputs = yolo.track(frame, stream=True)

    for output in outputs:
        classes_names = output.names
        for box in output.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = get_colours(cls)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 4)

                # display text on the bottom of the rectangle
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.3f}', (x1, y2),
                            cv2.FONT_HERSHEY_COMPLEX, 2, colour, 3)

    # show frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()
