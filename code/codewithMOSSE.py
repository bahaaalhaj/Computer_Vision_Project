import cv2
import csv
import time



def create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    elif hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    else:
        raise Exception("MOSSE tracker not available in your OpenCV build.")


# =========================
# Mouse callback variables
# =========================
drawing = False
ix, iy = -1, -1
current_box = None
frame_for_draw = None


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_box, frame_for_draw
    frame = param["frame"]
    add_fn = param["add"]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        frame_for_draw = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = frame.copy()
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 255), 2)
        frame_for_draw = img_copy

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = ix, iy
        w, h = abs(x - x0), abs(y - y0)
        x_min, y_min = min(x0, x), min(y0, y)

        if w > 0 and h > 0:
            add_fn(frame, (x_min, y_min, w, h))

        frame_for_draw = None


# =========================
# Main function
# =========================
def main():
    video_path = r"C:\Users\bahaa\OneDrive\Desktop\cvproject\inputvid\traffic.mp4"
    output_video = r"C:\Users\bahaa\OneDrive\Desktop\cvproject\outputvideo\output.avi"
    output_csv = r"C:\Users\bahaa\OneDrive\Desktop\cvproject\outputcsv\tracking_data.csv"
    text_color = (0, 255, 0)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video, fourcc, max(1, int(fps)), (W, H))

    # CSV file
    csv_file = open(output_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "frame", "object_id", "x", "y"])

    trackers = []
    boxes = []
    ids = []
    next_id = 0

    def add_tracker(frame, box):
        nonlocal next_id
        tracker = create_tracker()
        tracker.init(frame, box)
        trackers.append(tracker)
        boxes.append(box)
        ids.append(next_id)
        next_id += 1

    cv2.namedWindow("Video")
    mouse_param = {"frame": None, "add": add_tracker}
    cv2.setMouseCallback("Video", mouse_callback, mouse_param)

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        mouse_param["frame"] = frame
        timestamp = time.time()

        # Update trackers
        for i, tracker in enumerate(trackers):
            ok, box = tracker.update(frame)

            if ok:
                x, y, w, h = [int(v) for v in box]
                cx, cy = x + w // 2, y + h // 2

                writer.writerow([timestamp, frame_num, ids[i], cx, cy])
                cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 2)
                cv2.putText(frame, f"ID {ids[i]}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            else:
                cv2.putText(frame, f"ID {ids[i]} lost", (10, 30 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Draw temporary rectangle
        if frame_for_draw is not None:
            cv2.imshow("Video", frame_for_draw)
            out.write(frame_for_draw)
        else:
            cv2.imshow("Video", frame)
            out.write(frame)

        key = cv2.waitKey(100)& 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()