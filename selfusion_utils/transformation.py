import logging
import threading
import time
import numpy as np
import os
import cv2
import asyncio
from selfusion_utils.call_sdxlturbo import request_sdxlturbo
from neural_style_transfer.nst import generate_image_list
from yolo_v8_face.utils.video import Stream
from selfusion_utils.args import get_args
from selfusion_utils.leds import LedController

args, unknown = get_args()
if args.cam_device_number:
    logging.info("""You chose camera device no: {}""".format(args.cam_device_number))
    device = [args.cam_device_number]
else:
    device = None

WIDTH, HEIGHT = (1280, 1024)
FPS = args.fps
LONGER_DELAY_MS = max(int(args.gif_pause_sec * 1000), 1)
LOADING_DURATION_SEC = args.loading_duration_sec
WAIT_SEC = args.wait_sec


def led_sync_worker(app):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        led_controller = LedController()
        loop.run_until_complete(led_controller.connect())
        current_state = None
        while True:
            try:
                desired_state = "blink" if app.is_generating else "solid"
                if desired_state != current_state:
                    if desired_state == "blink":
                        loop.run_until_complete(led_controller.blink_white())
                    else:
                        loop.run_until_complete(led_controller.white())
                    current_state = desired_state
                time.sleep(1)  # check once per second
            except Exception as e:
                logging.info(f"LED error: {e}")
                break
    except Exception as e:
        logging.info(f"LED error: {e}")


# Global exception handler for uncaught exceptions in threads
def custom_excepthook(exc_info):
    """for any uncaught exception in threads to trigger systemd restart"""
    thread = threading.current_thread()
    logging.error(f"Uncaught exception in thread {thread.name}: {exc_info[1]}")  # Log the exception
    os._exit(1)


# Set the global exception handler for all threads
threading.excepthook = custom_excepthook


class Transformation:
    def __init__(self):
        self.frames = []
        self.gif_lock = threading.Lock()
        self.new_frames_event = threading.Event()
        self.selfie_lock = threading.Lock()
        self.selfie_ready_event = threading.Event()
        self.wait_screen = True
        self.selfie = None
        self.is_generating = False
        self.come_closer_screen = args.come_closer_screen
        self.stream = Stream(see_detection=args.see_detection,
                             available_devices=device,
                             face_size_threshold=args.face_size_threshold)


    def request_sdxlturbo_with_flag(self, result_container, stop_flag):
        request_sdxlturbo(self.selfie, result_container)
        if result_container.get('success', False):
            stop_flag.set()

    def processing_worker(self):
        sdxl_result = {}
        stop_flag = threading.Event()
        sdxl_start = time.time()
        sdxl_thread = threading.Thread(
            target=self.request_sdxlturbo_with_flag,
            args=(sdxl_result, stop_flag))
        sdxl_thread.start()

        logging.info("Generating NST images...")
        nst_images = generate_image_list(self.selfie, should_stop=stop_flag.is_set)
        sdxl_thread.join(timeout=LOADING_DURATION_SEC)

        if sdxl_result.get('success', False):
            logging.info(f"SDXLTurbo took {time.time() - sdxl_start} sec")
            gif_frames = [img for img in sdxl_result['images']]
        else:
            gif_frames = [np.array(img) for img in nst_images]

        with self.gif_lock:
            self.frames = gif_frames
            self.wait_screen = False
            self.new_frames_event.set()

    def selfie_capture_worker(self):
        self.selfie = self.stream.draw_boxes()
        logging.info("Selfie captured.")

        with self.selfie_lock:
            self.selfie_ready_event.set()

    def run(self):
        threading.Thread(target=led_sync_worker, args=(self,), daemon=True).start()
        self.is_generating = False
        next_selfie_time = 0
        loading_start_time = 0
        threading.Thread(target=self.selfie_capture_worker).start()
        while True:
            # If selfie ready and no processing running → start processing worker
            if self.selfie_ready_event.is_set() and not self.is_generating:
                threading.Thread(target=self.processing_worker).start()
                self.is_generating = True
                loading_start_time = time.time()
                self.selfie_ready_event.clear()

            # gif frames or noise
            with self.gif_lock:
                bounce_frames, delay_ms = self.prepare_bounce_frames()

            for i, frame in enumerate(bounce_frames):
                frame_copy = self.decorate_frame(frame, loading_start_time, LOADING_DURATION_SEC)
                canvas = self.create_canvas(frame_copy)

                cv2.imshow("TRANSFORMATION", canvas)
                # cv2.imshow("TRANSFORMATION", frame_resized)
                # cv2.imshow("TRANSFORMATION", frame_copy)

                is_turning_point = (i == 0 or i == len(bounce_frames) // 2 or i == len(bounce_frames) - 1)
                wait = LONGER_DELAY_MS if is_turning_point else delay_ms

                key = cv2.waitKey(wait)
                if key == 27:  # ESC to quit
                    cv2.destroyAllWindows()
                    return

                if self.new_frames_event.is_set():
                    self.is_generating = False
                    self.new_frames_event.clear()
                    # wait for 30 sec
                    next_selfie_time = time.time() + WAIT_SEC

                if not self.selfie_ready_event.is_set() and not self.is_generating and next_selfie_time != 0:
                    self.wait_screen = False
                    self.is_generating = False
                    logging.info(f"Waiting")

                    if time.time() >= next_selfie_time:
                        if self.come_closer_screen:
                            self.wait_screen = True
                        threading.Thread(target=self.selfie_capture_worker).start()
                        next_selfie_time = 0

        cv2.destroyAllWindows()

    def prepare_bounce_frames(self):
        if self.frames and not self.wait_screen:
            frames_bgr = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in self.frames]
            bounce_frames = frames_bgr + frames_bgr[-2::-1]
            delay_ms = max(int(1000 / FPS), 1)
        else:
            bounce_frames = [np.random.randint(0, 256, (HEIGHT, HEIGHT, 3), dtype=np.uint8)]
            delay_ms = 0
        return bounce_frames, delay_ms

    @staticmethod
    def create_canvas(frame):
        frame_resized = cv2.resize(frame, (HEIGHT, HEIGHT), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        x_offset = (WIDTH - HEIGHT) // 2
        y_offset = (HEIGHT - HEIGHT) // 2
        canvas[y_offset:y_offset + HEIGHT, x_offset:x_offset + HEIGHT] = frame_resized
        return canvas

    def decorate_frame(self,
                       frame,
                       loading_start_time,
                       loading_duration):
        frame_copy = frame.copy()
        if self.is_generating:
            progress = min(1.0, (time.time() - loading_start_time) / loading_duration)
            self.draw_loading_bar(frame_copy, progress)
        elif self.wait_screen:
            self.draw_come_closer(frame_copy)
        return frame_copy

    @staticmethod
    def draw_loading_bar(frame, progress, bar_width=400, bar_height=20):
        """loading bar with noise background"""
        # bar position
        x_center = frame.shape[1] // 2
        y_top = 60  # pixels from top

        x1 = x_center - bar_width // 2
        y1 = y_top
        x2 = x1 + bar_width
        y2 = y1 + bar_height

        # Full white bar background
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Shrinking noise overlay (from right to left)
        overlay_width = int(bar_width * (1 - progress))
        if overlay_width > 0:
            noise = np.random.randint(0, 256, (bar_height, overlay_width, 3), dtype=np.uint8)
            frame[y1:y2, x2 - overlay_width:x2] = noise

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"LOADING YOUR TRANSFORMATION"
        text_scale = 0.6
        text_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, text_scale, text_thickness)
        text_x = x_center - text_size[0] // 2
        text_y = y1 - 15  # 15 pixels above bar

        cv2.putText(frame, text, (text_x, text_y), font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    @staticmethod
    def draw_come_closer(frame):
        x_center = frame.shape[1] // 2
        y_center = frame.shape[0] // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"COME CLOSER"
        text_scale = 2
        text_thickness = 3
        text_size, _ = cv2.getTextSize(text, font, text_scale, text_thickness)
        text_x = x_center - text_size[0] // 2

        cv2.putText(frame, text, (text_x, y_center), font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)