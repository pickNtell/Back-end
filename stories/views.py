import os
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.conf import settings

from pathlib import Path
from openai import OpenAI
import requests
from langdetect import detect, LangDetectException
from io import BytesIO

from elevenlabs import play
from elevenlabs.client import ElevenLabs
from elevenlabs import save

import base64

import json
import time
import uuid 


from dotenv import load_dotenv
load_dotenv('config.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STABLE_DIFFUSION_API_KEY = os.getenv('STABLE_DIFFUSION_API_KEY')
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')

# Main page path (Typeing informations about fairy tale user want)
def home(request):
    return render(request, '1start.html')

# Second page path (Generate and Show the fairy tale)
def generate_story(request):
    if request.method == 'POST':
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        genre = request.POST.get('genre')
        characters = request.POST.getlist('characters[]')
        details = request.POST.get('details', False)
        print(age, gender, genre, characters, details)
    ### 1. Setting prompt ###
        system_input = "You are a professional fairy tale writer. Create a creative fairy tale tailored to  the child's age, gender, and preferred genre. Be sure to use vocabulary appropriate for the child's age and keep the story length suitable. Provide both a Korean and an English version of the story. ex) korean:\n ~~ \n english:\n ~~"
        user_input = f"The child's age is {age}, gender is {gender}, and the desired genre is {genre}. Please create a fairy tale based on these preferences."
        if characters is not None:
            user_input += f"The story must include characters: {', '.join(characters)}. "
        if details:
            user_input += f"Include detailed descriptions: {', '.join(details)}."
    ##########################
        
        
    ### 2. Generate a scenario ###
        scenario = generate_scenario(system_input, user_input)
        # print(scenario)
    ##############################
        
        
    ### 3. Post-processing the scenario ###
        # Separate paragraphs
        paragraphs = scenario.split("\n\n")
        # Separate the json file by languages
        story_json = {"Korean": [], "English": []}
        k = -1
        for idx, para in enumerate(paragraphs):
            if not para.strip():  # 빈 문자열 또는 공백 문자열 확인
                print(f"Empty paragraph detected at index {idx}. Skipping.")
                continue
            try:
                lang = detect(para)
                if lang == "ko":
                    story_json["Korean"].append({"scene": idx + 1, "content": para})
                    k += 1
                elif lang == "en":
                    story_json["English"].append({"scene": idx - k, "content": para})
            except LangDetectException as e:
                print(f"LangDetectException for paragraph {idx}: {e}")
        
        ### For debugging ###
        # with open("./story.json", "r", encoding="utf-8") as file:
        #     story_json = json.load(file)
        #####################
    ########################################
    
    
    ### 4. Generate images using the scenario above ###
        output_format = "jpeg"
        aspect_ratio = "16:9"
        generated_images = generate_images_from_json(story_json['English'], output_format, aspect_ratio)
        print(generated_images)
        
        ### For debugging ###
        # with open("./image.json", "r", encoding="utf-8") as file:
        #     generated_images = json.load(file)
        #####################
    ###################################################


    ### 5. Generate TTS using the scenario above ###
        generated_TTS = generate_audio_save(story_json['Korean'])
        ### For debugging ###
        # with open("./audio.json", "r", encoding="utf-8") as file:
        #     generated_TTS = json.load(file)
        #####################
    ###################################################


    ### Create a context that combines Scenario, Images, Audios ###
        pages = []
        for idx, (scene, img, tts) in enumerate(zip(story_json["Korean"], generated_images, generated_TTS)):
            pages.append({
                "text": scene['content'],
                "image": img['file'],
                "audio": tts['file'],
                "scene": idx + 1,
            })
        context = {
            "pages": pages,
        }
    ################################################################
                
                
                
        # Send both a html and context(scenario+images+TTS)
        return render(request, '2story.html', context)

    # If the function recieves nothing, throw the 400 error
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def generate_scenario(system_input, user_input, model="gpt-4o-mini", temperature=1.15, max_tokens=150):
    try:
        client = OpenAI()

        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        scenario = response.choices[0].message.content
        return scenario
    
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


def send_generation_request(
    host,
    params,
    files = None
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABLE_DIFFUSION_API_KEY}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def send_async_generation_request(
    host,
    params,
    files = None
):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABLE_DIFFUSSION_API_KEY}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    # Process async response
    response_dict = json.loads(response.text)
    generation_id = response_dict.get("id", None)
    assert generation_id is not None, "Expected id in response"

    # Loop until result or timeout
    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    status_code = 202
    while status_code == 202:
        print(f"Polling results at https://api.stability.ai/v2beta/results/{generation_id}")
        response = requests.get(
            f"https://api.stability.ai/v2beta/results/{generation_id}",
            headers={
                **headers,
                "Accept": "*/*"
            },
        )

        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        status_code = response.status_code
        time.sleep(10)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

    return response

def generate_image_from_prompt(prompt, negative_prompt="", aspect_ratio="16:9", seed=0, output_format="jpeg"):
    
    host = "https://api.stability.ai/v2beta/stable-image/generate/core"
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "output_format": output_format
    }

    # 요청 전송
    response = send_generation_request(host, params)

    # 응답 데이터 처리
    output_image = response.content
    finish_reason = response.headers.get("finish-reason")
    response_seed = response.headers.get("seed", str(uuid.uuid4()))  # 없으면 UUID 사용

    # NSFW 필터링 확인
    if finish_reason == "CONTENT_FILTERED":
        raise Warning("Generation failed due to NSFW content.")

    # 저장 디렉토리 설정
    images_dir = os.path.join(settings.MEDIA_ROOT, 'images')
    os.makedirs(images_dir, exist_ok=True)  # 디렉토리 생성 (존재하면 무시)

    # 파일 이름 및 경로 설정
    generated_filename = f"generated_{response_seed}.{output_format}"
    file_path = os.path.join(images_dir, generated_filename)

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(output_image)

    print(f"Image saved as: {file_path}")
    return file_path


def generate_images_from_json(story_scenes, output_format="jpeg", aspect_ratio="16:9"):

    generated_files = []

    for scene in story_scenes:
        scene_number = scene["scene"]
        prompt = scene["content"]
        print(f"Generating image for Scene {scene_number}...")

        try:
            # 이미지 생성 및 절대 경로 반환
            absolute_path = generate_image_from_prompt(
                prompt=prompt,
                negative_prompt="",
                aspect_ratio=aspect_ratio,
                seed=0,
                output_format=output_format
            )

            # 상대 경로로 변환 ### 유저에게 127.0.0.1/media/images/generated_1390576109.jpeg 이런식으로 보내지게~
            relative_path = Path(absolute_path).relative_to(settings.MEDIA_ROOT)
            web_path = f"{settings.MEDIA_URL}{relative_path}"

            generated_files.append({"scene": scene_number, "file": web_path})
        except Warning as w:
            print(f"Scene {scene_number}: {w}")
        except Exception as e:
            print(f"Scene {scene_number}: {e}")

    print("All images generated.")
    return generated_files

def generate_TTS(scene_number, story):
    client = ElevenLabs(
        api_key = ELEVEN_LABS_API_KEY
    )
            # 텍스트로 음성 생성
    audio = client.generate(
            text=story,
            voice="Jessica",
            model="eleven_multilingual_v2"
    )

    # 저장 디렉토리 설정
    audio_dir = os.path.join(settings.MEDIA_ROOT, 'audios')
    os.makedirs(audio_dir, exist_ok=True)  # 디렉토리 생성 (존재하면 무시)

    # 파일 이름 및 경로 설정
    random_uuid = uuid.uuid4()
    generated_filename = f"generated_{scene_number}_{random_uuid}.mp3"
    file_path = os.path.join(audio_dir, generated_filename)

    save(audio, file_path)

    
    print(f"Audio content written to file: {file_path}")

    return file_path


def generate_audio_save(story_scenes):
    
    generated_files = []

    for scene in story_scenes:
        scene_number = scene["scene"]
        prompt = scene["content"]
        print(f"Generating TTS for Scene {scene_number}...")
        try:
            # 이미지 생성 및 절대 경로 반환
            absolute_path = generate_TTS(scene_number, prompt)

            # 상대 경로로 변환 ### 유저에게 127.0.0.1/media/audios/generated_1390576109.mp3 이런식으로 보내지게~
            relative_path = Path(absolute_path).relative_to(settings.MEDIA_ROOT)
            web_path = f"{settings.MEDIA_URL}{relative_path}"

            generated_files.append({"scene": scene_number, "file": web_path})
        except Warning as w:
            print(f"Scene {scene_number}: {w}")
        except Exception as e:
            print(f"Scene {scene_number}: {e}")

    print("All TTS generated.")
    return generated_files



