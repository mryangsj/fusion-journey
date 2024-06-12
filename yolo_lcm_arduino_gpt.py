import os
import threading

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import datetime as dt
import logging
from typing import Dict, Any
import requests
import serial
import time
from dotenv import load_dotenv
from openai import OpenAI
from pynput import keyboard
import platform
from pythonosc import udp_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure torch
torch.set_num_threads(8)

# Global variables
LANDMARK = 'Duomo'

CITY = 'Milan'

LCM_PROMPT = None
QUIT_FLAG = False
RESET_FLAG = False
GLOCK = threading.Lock()

BACK_IMG = cv2.resize(cv2.imread("image/Duomo_Milan.png"), (512, 512))
CANNY = np.zeros((512, 512, 3), dtype=np.uint8)

CANVAS = np.zeros((512 * 2, 512 * 2, 3), dtype=np.uint8)
CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)


class ArduinoException(Exception):
    pass


class Arduino(threading.Thread):
    def __init__(self, serial_port, baud_rate=9600, timeout=1, prompt_callback=None, verbose=True):
        super().__init__()
        self.port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.verbose = verbose
        self.serial_address = self._determine_serial_address()
        self.serial = None
        self.sensor_humidity = 0.0
        self.sensor_temperature = 0.0
        self.initial_sensor_humidity = 0.0
        self.initial_sensor_temperature = 0.0
        self.sensor_changed = False
        self.SENSOR_CHANGE_THRESHOLD = 2
        self.prompt_callback = prompt_callback
        self.last_callback_time = 0

    def _determine_serial_address(self):
        os_name = platform.system()
        if os_name == "Darwin":
            return f"/dev/cu.usbmodem{self.port}"
        elif os_name == "Windows":
            return f"COM{self.port}"
        elif os_name == "Linux":
            return f"/dev/ttyACM{self.port}"
        else:
            raise ArduinoException(f"Unsupported operating system: {os_name}")

    def _create_serial_connection(self):
        try:
            self.serial = serial.Serial(self.serial_address, self.baud_rate, timeout=self.timeout)
            for _ in range(5): self.serial.readline()  # Clear the buffer
        except serial.SerialException as e:
            raise ArduinoException(f"Failed to create serial connection: {e}")

    def _close_serial_connection(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def _initialize_sensor_data(self):
        if not self.serial:
            self._create_serial_connection()

        self.initial_sensor_humidity, self.initial_sensor_temperature = self._read_sensor_data()

    def _read_sensor_data(self):
        if self.serial is None:
            self._create_serial_connection()

        for _ in range(5):
            try:
                data = self.serial.readline().decode().strip()
                if data:
                    parts = data.split(',')
                    if len(parts) == 2:
                        return float(parts[0]), float(parts[1])
            except (serial.SerialException, ValueError) as e:
                logging.warning(f"Error when reading sensor data: {e}")
                time.sleep(1)  # Wait a bit before retrying

        raise ArduinoException("Failed to read sensor data after 5 attempts")

    def get_sensor_data(self):
        return self.sensor_humidity, self.sensor_temperature

    def set_sensor_change_threshold(self, threshold):
        self.SENSOR_CHANGE_THRESHOLD = threshold

    def run(self):
        global LANDMARK, CITY, LCM_PROMPT, RESET_FLAG, QUIT_FLAG

        self._initialize_sensor_data()

        min_callback_interval = 60  # 最短调用间隔为1分钟
        max_callback_interval = 30 * 60  # 最大调用间隔为30分钟

        while not QUIT_FLAG:
            if RESET_FLAG:
                self.reset()
                RESET_FLAG = False
            try:
                self.sensor_humidity, self.sensor_temperature = self._read_sensor_data()
                humidity_change = abs(self.sensor_humidity - self.initial_sensor_humidity)
                temperature_change = abs(self.sensor_temperature - self.initial_sensor_temperature)
                if self.verbose:
                    print(f"Humidity: {self.sensor_humidity:.0f}% (Δ: {humidity_change:.0f}), "
                          f"Temperature: {self.sensor_temperature:.0f}°C (Δ: {temperature_change:.0f})")

                current_time = time.time()
                time_since_last_callback = current_time - self.last_callback_time

                if humidity_change >= self.SENSOR_CHANGE_THRESHOLD or temperature_change >= self.SENSOR_CHANGE_THRESHOLD:
                    self.initial_sensor_humidity = self.sensor_humidity
                    self.initial_sensor_temperature = self.sensor_temperature

                    if time_since_last_callback >= min_callback_interval:
                        if self.prompt_callback:
                            with GLOCK:
                                self.last_callback_time = current_time
                                LCM_PROMPT = self.prompt_callback(LANDMARK, CITY)

                    if self.verbose: print("Sensor change detected.")
                elif time_since_last_callback >= max_callback_interval:
                    if self.prompt_callback:
                        with GLOCK:
                            self.last_callback_time = current_time
                            LCM_PROMPT = self.prompt_callback(LANDMARK, CITY)

                    if self.verbose: print("Maximum callback interval reached.")

            except ArduinoException as e:
                logging.error(f"Arduino error: {e}")

            time.sleep(1)

        self.stop()

    def stop(self):
        self.running = False
        self._close_serial_connection()

    def reset(self):
        self.last_callback_time = 0


class PromptGenerator:
    API_KEY_ENV_NAMES = {
        "openai": "OPENAI_API_KEY",
        "weather": "OPENWEATHERMAP_API_KEY"
    }

    WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    KELVIN_TO_CELSIUS_OFFSET = 273.15
    GPT_MODEL = "gpt-4o"
    GPT_DEFAULT_TOKENS = 70

    def __init__(self):
        self.openai_api_key = self._load_env_variable(PromptGenerator.API_KEY_ENV_NAMES["openai"])
        self.weather_api_key = self._load_env_variable(PromptGenerator.API_KEY_ENV_NAMES["weather"])
        self.gpt = OpenAI(api_key=self.openai_api_key)

    @staticmethod
    def _load_env_variable(env_name: str) -> str:
        load_dotenv()
        key = os.getenv(env_name)
        if not key:
            logging.error(f"Environment variable for {env_name} is not set.")
            raise ValueError(f"Environment variable for {env_name} is not set.")
        return key

    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        return kelvin - PromptGenerator.KELVIN_TO_CELSIUS_OFFSET

    @staticmethod
    def format_time(timestamp: int, timezone: int) -> str:
        return dt.datetime.fromtimestamp(timestamp + timezone).strftime('%H:%M:%S')

    def get_weather_data(self, city: str = "Milan") -> Dict[str, Any]:
        url = f"{PromptGenerator.WEATHER_BASE_URL}?q={city}&appid={self.weather_api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error occurred while fetching weather data: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logging.error(f"Request Exception occurred: {e}")
            raise

    def get_prompt(self, landmark: str = "Duomo", city: str = "Milan", verbose=True) -> str:
        try:
            weather_data = self.get_weather_data(city)

            timezone_offset = weather_data['timezone']
            utc_now = dt.datetime.now(dt.timezone.utc)
            local_time = utc_now + dt.timedelta(seconds=timezone_offset)
            current_time = local_time.strftime('%H:%M:%S')

            temp_celsius = self.kelvin_to_celsius(weather_data['main']['temp'])
            description = weather_data['weather'][0]['description']

            local_sunrise = self.format_time(weather_data['sys']['sunrise'], timezone_offset)
            local_sunset = self.format_time(weather_data['sys']['sunset'], timezone_offset)

            system_content, user_content = self.build_prompt_content(
                landmark,
                description,
                temp_celsius,
                current_time,
                local_sunrise,
                local_sunset,
            )

            return self.query_gpt(system_content, user_content, verbose)

        except Exception as e:
            logging.error(f"Error in get_prompt: {e}")
            raise

    @staticmethod
    def build_prompt_content(
            landmark: str,
            weather_description: str,
            temp_celsius: float,
            current_time,
            sunrise_time,
            sunset_time
    ):
        system_message = (
            f"You are a creative assistant specialized in generating informative prompts for image generative models "
            f"like Stable Diffusion, DALL-E, etc. You are tasked with creating a prompt for a tourism scene showing "
            f"the most famous landmark with the current weather conditions."
        )

        user_message = (
            f"Write a one sentence prompt showing the most famous tourism scene in {landmark} with the weather being "
            f"{weather_description} at {temp_celsius:.1f}°C, observed at time {current_time}. Please accurately "
            f"reflect the scene within the specified time range, given that the sun rises at {sunrise_time} and sun "
            f"sets at {sunset_time}, but don't show the exact time and temperature in the prompt. Append suitable "
            f"descriptive labels you think that may appear in the scene at the end, e.g. 'animals, sheep, birds, "
            f"cinematic, scenery, 8k', etc. Here is an example of expected prompt: 'Duomo di Milano under the early "
            f"morning sky with broken clouds, the iconic cathedral subtly illuminated by streetlights and the quiet "
            f"ambiance of the city at predawn hour with a few nocturnal birds in flight. cinematic, serene, "
            f"atmospheric, 8k.'"
        )

        return system_message, user_message

    def query_gpt(self, system_content: str, user_content: str, verbose=True) -> str:
        logging.debug(f"Asking GPT: {user_content}")
        try:
            completion = self.gpt.chat.completions.create(
                model=PromptGenerator.GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=PromptGenerator.GPT_DEFAULT_TOKENS
            )
            gpt_response = completion.choices[0].message.content

            if verbose:
                print(f"Ask GPT: {user_content}")
                print(f"GPT Response: {gpt_response}")

            return gpt_response
        except Exception as e:
            logging.error(f"Error occurred while generating prompt: {e}")
            raise


class YoloSeg(threading.Thread):
    YOLO_PATH = "yolov8n-seg.torchscript"
    YOLO_INPUT_SIZE = 640
    RESULT_SIZE = 512
    OSC_PORT = 12000

    def __init__(self, camera_id=1):
        super().__init__()
        self.yolo = YOLO(YoloSeg.YOLO_PATH)
        self.cap = cv2.VideoCapture(camera_id)

        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_size = min(self.cap_width, self.cap_height)

        self.frame_raw = np.zeros((self.cap_width, self.cap_height, 3), dtype=np.uint8)
        self.frame_crop = np.zeros((self.cap_size, self.cap_size, 3), dtype=np.uint8)
        self.frame_crop_resized = np.zeros((YoloSeg.YOLO_INPUT_SIZE, YoloSeg.YOLO_INPUT_SIZE, 3), dtype=np.uint8)
        self.frame_crop_x = (self.cap_width - self.cap_size) // 2
        self.frame_crop_y = (self.cap_height - self.cap_size) // 2

        self.result = None
        self.result_img = np.zeros((YoloSeg.RESULT_SIZE, YoloSeg.RESULT_SIZE, 3), dtype=np.uint8)
        self.people_mask = None

        self.client = udp_client.SimpleUDPClient("127.0.0.1", YoloSeg.OSC_PORT)

    def stop(self):
        self.cap.release()

    def run(self):
        global BACK_IMG, CANVAS, CANNY, QUIT_FLAG
        while not QUIT_FLAG:
            ret, self.frame_raw = self.cap.read()

            self.frame_crop = self.frame_raw[self.frame_crop_y:self.frame_crop_y + self.cap_size, self.frame_crop_x:self.frame_crop_x + self.cap_size]
            self.frame_crop_resized = cv2.resize(self.frame_crop, (YoloSeg.RESULT_SIZE, YoloSeg.RESULT_SIZE))

            self.result = self.yolo(self.frame_crop_resized, verbose=False)[0]

            self.result_img = cv2.cvtColor(self.result.plot(), cv2.COLOR_RGB2BGR)
            with GLOCK:
                CANVAS[0:YoloSeg.RESULT_SIZE, 0:YoloSeg.RESULT_SIZE] = cv2.resize(self.result_img, (YoloSeg.RESULT_SIZE, YoloSeg.RESULT_SIZE))

            if self.result.masks is not None:
                # Filter the person class
                masks = self.result.masks.data
                boxes = self.result.boxes.data
                labels = boxes[:, 5].int()
                people_masks = masks[labels == 0]

                # Resize the person masks to the expected output size
                people_masks = people_masks.unsqueeze(1)
                people_masks = torch.nn.functional.interpolate(
                    people_masks,
                    (YoloSeg.RESULT_SIZE, YoloSeg.RESULT_SIZE),
                    mode="nearest"
                ).squeeze(1)

                # Create a person mask
                self.people_mask = torch.sum(people_masks, dim=0)

                # Mask the background image
                img_comp = np.copy(BACK_IMG)
                img_comp[self.people_mask > 0] = 0

                # Canny edge detection
                canny_local = cv2.Canny(img_comp, 100, 200)
                with GLOCK:
                    CANNY = np.stack([canny_local] * 3, axis=-1)
                    CANVAS[YoloSeg.RESULT_SIZE:, 0:YoloSeg.RESULT_SIZE] = CANNY

                # Compute the center of each person mask and send OSC message
                people_masks_np = people_masks.cpu().numpy()
                for mask in people_masks_np:
                    coords = np.argwhere(mask > 0)
                    if coords.size > 0:
                        center_x = np.mean(coords[:, 1])
                        center_y = np.mean(coords[:, 0])
                        self.client.send_message("/coordinates", [int(center_x), int(center_y)])

        self.stop()


class LCM(threading.Thread):
    LCM_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    LCM_PATH = "models/LCM_Dreamshaper_v7"
    CONTROL_NET_PATH = "models/control_v11p_sd15_canny"

    NUM_INFERENCE_STEPS = 4
    GUIDANCE_SCALE = 4.0
    CONTROLNET_CONDITIONING_SCALE = 1.0

    LCM_WIDTH = 512
    LCM_HEIGHT = 512
    SEED = 19

    LCM_NEGATIVE_PROMPT = "bad anatomy, deformed, ugly, disfigured, low quality, bad quality, sketches"

    def __init__(self):
        super().__init__()
        print(f"Using device: {LCM.LCM_DEVICE} for LCM.")

        self.ctrl_net = ControlNetModel.from_pretrained(LCM.CONTROL_NET_PATH, use_safetensors=True)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            LCM.LCM_PATH,
            controlnet=self.ctrl_net,
            safety_checker=None
        ).to(LCM.LCM_DEVICE)
        if LCM.LCM_DEVICE == "mps": self.pipe.enable_attention_slicing()

    def run(self):
        global LCM_PROMPT, CANNY, QUIT_FLAG
        while not QUIT_FLAG:
            if LCM_PROMPT is not None:
                image = self.pipe(
                    prompt                        = LCM_PROMPT,
                    negative_prompt               = LCM.LCM_NEGATIVE_PROMPT,
                    image                         = Image.fromarray(CANNY),
                    width                         = LCM.LCM_WIDTH,
                    height                        = LCM.LCM_HEIGHT,
                    num_inference_steps           = LCM.NUM_INFERENCE_STEPS,
                    guidance_scale                = LCM.GUIDANCE_SCALE,
                    controlnet_conditioning_scale = LCM.CONTROLNET_CONDITIONING_SCALE,
                    generator                     = torch.Generator(device=LCM.LCM_DEVICE).manual_seed(LCM.SEED),
                    guess_mode                    = True,
                    output_type                   = "np",
                    verbose                       = False,
                ).images[0]

                CANVAS[512:, 512:] = (image * 255.0).astype(np.uint8)
            else:
                time.sleep(1)


# # Crop the background image to a square
# img_back_width, img_back_height = img_back.shape[:2]
# img_back_size = min(img_back_width, img_back_height)
# img_back_crop_x, img_back_crop_y = (img_back_width - img_back_size) // 2, (img_back_height - img_back_size) // 2
# img_back = img_back[img_back_crop_y:img_back_crop_y + img_back_size, img_back_crop_x:img_back_crop_x + img_back_size]


def display_thread():
    global CANVAS, QUIT_FLAG
    while not QUIT_FLAG:
        canvas_local = cv2.cvtColor(CANVAS, cv2.COLOR_RGB2BGR)
        cv2.imshow("Results", canvas_local)
        cv2.waitKey(int(1000/30))
    cv2.destroyAllWindows()


def on_press(key):
    global LANDMARK, CITY, BACK_IMG, RESET_FLAG, QUIT_FLAG
    if key == keyboard.Key.esc:
        QUIT_FLAG = True
        return False
    if key.char == "1":
        LANDMARK = "Duomo"
        CITY = "Milan"
        BACK_IMG = cv2.resize(cv2.imread("image/Duomo_Milan.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "2":
        LANDMARK = "Funes"
        CITY = "Bolzano"
        BACK_IMG = cv2.resize(cv2.imread("image/Funes_Bolzano.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "3":
        LANDMARK = " Egyptian Pyramids"
        CITY = "Cairo"
        BACK_IMG = cv2.resize(cv2.imread("image/Pyramids_Cairo.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "4":
        LANDMARK = "Forbidden City"
        CITY = "Beijing"
        BACK_IMG = cv2.resize(cv2.imread("image/ForbiddenCity_Beijing.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "5":
        LANDMARK = "Times Square"
        CITY = "New York"
        BACK_IMG = cv2.resize(cv2.imread("image/TimesSquare_NewYork.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "6":
        LANDMARK = "Eiffel Tower"
        CITY = "Paris"
        BACK_IMG = cv2.resize(cv2.imread("image/EiffelTower_Paris.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "7":
        LANDMARK = "Acropolis"
        CITY = "Athens"
        BACK_IMG = cv2.resize(cv2.imread("image/Acropolis_Athens.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "8":
        LANDMARK = "Opera House"
        CITY = "Sydney"
        BACK_IMG = cv2.resize(cv2.imread("image/OperaHouse_Sydney.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True
    elif key.char == "9":
        LANDMARK = "Yosemite"
        CITY = "California"
        BACK_IMG = cv2.resize(cv2.imread("image/Yosemite_California.png"), (512, 512))
        CANVAS[0:512, 512:1024] = cv2.cvtColor(BACK_IMG, cv2.COLOR_BGR2RGB)
        RESET_FLAG = True


def main():
    # %% Arduino setup
    arduino_serial_port = 1101

    try:
        key_listener = keyboard.Listener(on_press=on_press)
        key_listener.start()

        prompt_generator = PromptGenerator()
        arduino_thread = Arduino(
            serial_port=arduino_serial_port,
            prompt_callback=prompt_generator.get_prompt
        )
        arduino_thread.start()

        yolo_thread = YoloSeg(2)
        yolo_thread.start()

        lcm_thread = LCM()
        lcm_thread.start()

        display_thread()

        lcm_thread.join()
        yolo_thread.join()
        arduino_thread.join()
        key_listener.join()

    except KeyboardInterrupt:
        logging.info("Program terminated by user")
    except Exception as e:
        logging.error(f'Unexpected error: {e}')
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
