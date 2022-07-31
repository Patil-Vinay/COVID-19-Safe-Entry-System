# Screen at the entrance, drawn with OpenCV.
# Shows the camera, what the person has to do next and how many people are inside.
import os

import cv2
import numpy as np

WIDTH, HEIGHT = 800, 480  # 7 inch Pi display, other screens are scaled

# colours are BGR
WHITE = (255, 255, 255)
GREY = (140, 140, 140)
GREEN = (90, 200, 70)
YELLOW = (0, 200, 255)
RED = (60, 60, 230)
BACKGROUND = (30, 30, 30)

STEPS = ("Mask", "Temperature", "Sanitizer")


def screen_available():
    # OpenCV can only open a window when the desktop is running
    return os.name == "nt" or bool(os.environ.get("DISPLAY"))


class EntranceScreen:
    def __init__(self, max_people, enabled=True, fullscreen=True):
        self.max_people = max_people
        self.enabled = enabled
        self.inside = 0
        self.frame = None
        self.reset()
        if enabled:
            cv2.namedWindow("Sentinel", cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty("Sentinel", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def reset(self):
        self.message = "Please stand at the entrance"
        self.color = GREY
        self.steps = {name: None for name in STEPS}
        self.box = None

    def set_step(self, name, text, color):
        self.steps[name] = (text, color)

    def set_frame(self, frame, box=None, label="", color=WHITE):
        self.frame = frame
        self.box = None if box is None else (box, label, color)

    def refresh(self):
        if not self.enabled:
            return
        cv2.imshow("Sentinel", self.draw())
        if cv2.waitKey(1) & 0xFF == 27:  # Esc closes the program
            raise KeyboardInterrupt

    def draw(self):
        img = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)

        # camera on the left, 4:3 like the Pi camera
        cam_w, cam_h = 512, 384
        if self.frame is None:
            cv2.rectangle(img, (16, 16), (16 + cam_w, 16 + cam_h), GREY, 1)
            put_text(img, "No camera", (200, 210), 0.8, GREY)
        else:
            h, w = self.frame.shape[:2]
            cam = cv2.resize(self.frame, (cam_w, cam_h))
            if self.box is not None:
                (x1, y1, x2, y2), label, color = self.box
                sx, sy = cam_w / w, cam_h / h
                p1 = (int(x1 * sx), int(y1 * sy))
                p2 = (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(cam, p1, p2, color, 3)
                if label:
                    put_text(cam, label, (p1[0], max(25, p1[1] - 10)), 0.8, color, 2)
            img[16:16 + cam_h, 16:16 + cam_w] = cam

        # right side: title, people count and the checks
        x = 548
        put_text(img, "THE SENTINEL", (x, 50), 0.9, WHITE, 2)
        put_text(img, "COVID-19 safe entry", (x, 78), 0.55, GREY)

        put_text(img, "People inside", (x, 130), 0.6, GREY)
        full = self.inside >= self.max_people
        put_text(img, "%d / %d" % (self.inside, self.max_people), (x, 180), 1.5,
                 RED if full else WHITE, 3)

        y = 240
        for name in STEPS:
            state = self.steps[name]
            text, color = state if state else ("-", GREY)
            cv2.circle(img, (x + 8, y - 6), 7, color, -1)
            put_text(img, name, (x + 26, y), 0.6, WHITE)
            put_text(img, text, (x + 26, y + 24), 0.55, color)
            y += 56

        # message bar at the bottom
        cv2.rectangle(img, (16, 416), (WIDTH - 16, HEIGHT - 16), self.color, -1)
        put_text(img, self.message, (32, 452), 0.9, (20, 20, 20), 2, max_width=WIDTH - 64)
        return img


def put_text(img, text, pos, scale, color, thickness=1, max_width=None):
    # makes the text smaller if it does not fit
    if max_width:
        while scale > 0.4 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_width:
            scale -= 0.05
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
