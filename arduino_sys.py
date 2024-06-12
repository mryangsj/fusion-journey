import datetime as dt
import logging
import os
from typing import Dict, Any

import requests
import serial
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
CITY = 'Paris'

ser = serial.Serial('/dev/cu.usbmodem1401', 9600, timeout=1)  # macOS 示例


def kelvin_to_celsius(kelvin: float) -> float:
    return kelvin - 273.15


def get_weather_data(city: str) -> Dict[str, Any]:
    url = f'{BASE_URL}?q={city}&appid={API_KEY}'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def generate_prompt(weather_data: Dict[str, Any]) -> str:
    current_time = dt.datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S')
    temp_kelvin = weather_data['main']['temp']
    temp_celsius = kelvin_to_celsius(temp_kelvin)
    humidity = weather_data['main']['humidity']
    description = weather_data['weather'][0]['description']
    local_sunrise = dt.datetime.fromtimestamp(weather_data['sys']['sunrise'] + weather_data['timezone']).strftime('%H:%M:%S')
    local_sunset = dt.datetime.fromtimestamp(weather_data['sys']['sunset'] + weather_data['timezone']).strftime('%H:%M:%S')

    logging.info(f'Current time: {formatted_time}')
    logging.info(f'Description in {CITY}: {description}')
    logging.info(f'Temperature in {CITY}: {temp_celsius:.2f}°C')
    logging.info(f'Humidity in {CITY}: {humidity}%')
    logging.info(f'Sun rise in {CITY} at {local_sunrise} local time.')
    logging.info(f'Sun set in {CITY} at {local_sunset} local time.')

    prompt = f"""Write a one sentence prompt showing the most famous tourism scene in {CITY} with the weather being {description} at time {formatted_time}. Please accurately reflect the scene within the specified time range, given that the sun rises at {local_sunrise} and sun sets at {local_sunset}, but don't show the exact time in the prompt. Append suitable descriptive labels you think that may appear in the scene at the end, e.g. 'animals: sheep, birds, cinematic, scenery, 8k' This is an example of expected prompt: 'Duomo di Milano under the early morning sky with broken clouds, the iconic cathedral subtly illuminated by streetlights and the quiet ambiance of the city at predawn hour with a few nocturnal birds in flight. cinematic, serene, atmospheric, 8k.'"""

    print(f'Ask GPT: {prompt}')

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a creative assistant specialized in generating informative prompts \
            for image generative models like Stable Diffusion."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=70
    )

    return completion.choices[0].message.content


def read_from_arduino():
    data = ser.readline().decode().strip()
    if data:  # 确保读取到的数据非空
        parts = data.split(',')
        if len(parts) == 2:
            try:
                humidity = int(parts[0])
                temperature = int(parts[1])
                return humidity, temperature
            except ValueError:
                logging.error(f"Error parsing data: {data}")
    return None, None


def monitor_changes(interval=1, threshold=2, min_call_interval=60, max_call_interval=1800):
    initial_read = read_from_arduino()
    attempt_count = 0

    while initial_read == (None, None) and attempt_count < 5:
        logging.warning("Failed to get initial reading. Trying again in 1 second...")
        time.sleep(1)
        initial_read = read_from_arduino()
        attempt_count += 1

    if initial_read == (None, None):
        logging.error("Failed to get initial reading.")
        return

    last_call_time = 0
    last_significant_change_time = time.time()

    while True:
        humidity, temperature = read_from_arduino()
        if humidity is not None and temperature is not None and initial_read is not None:
            last_hum, last_temp = initial_read
            temp_change = abs(temperature - last_temp)
            hum_change = abs(humidity - last_hum)
            logging.info(f"Readings: Temperature = {temperature}°C (Change: {temp_change}), Humidity = {humidity}% (Change: {hum_change})")

            current_time = time.time()
            if (temp_change > threshold or hum_change > threshold) and (current_time - last_call_time >= min_call_interval):
                logging.info("Significant change detected!")
                weather_data = get_weather_data(CITY)
                prompt = generate_prompt(weather_data)
                print(f"GPT Response: {prompt}")
                initial_read = (humidity, temperature)
                last_call_time = current_time
                last_significant_change_time = current_time
            elif current_time - last_significant_change_time >= max_call_interval:
                logging.info("No significant change detected in the past 30 minutes. Forcing a call.")
                weather_data = get_weather_data(CITY)
                prompt = generate_prompt(weather_data)
                print(f"GPT Response: {prompt}")
                last_call_time = current_time
                last_significant_change_time = current_time

        time.sleep(interval)


def main():
    try:
        monitor_changes(interval=1, threshold=2)
    except KeyboardInterrupt:
        logging.info("Program terminated by user")
    except Exception as e:
        logging.error(f'Unexpected error: {e}')
    finally:
        ser.close()


if __name__ == "__main__":
    main()