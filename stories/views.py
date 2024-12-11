import os
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from openai import OpenAI
import requests
from langdetect import detect
from io import BytesIO
import json
import time

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STABLE_DIFFUSION_API_KEY = os.getenv('STABLE_DIFFUSION_API_KEY')
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')

# main page
def home(request):
    return render(request, '1start.html')

# second page
def generate_story(request):
    if request.method == 'POST':
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        genre = request.POST.get('genre')
        characters = request.POST.getlist('characters[]')
        details = request.POST.get('details', False)
    
        
        system_input = "You are a professional fairy tale writer. Create a creative fairy tale tailored to the child's age, gender, and preferred genre. Be sure to use vocabulary appropriate for the child's age and keep the story length suitable. Provide both a Korean and an English version of the story. ex) korean: \n ~~, english:\n"
        user_input = f"The child's age is {age}, gender is {gender}, and the desired genre is {genre}. Please create a fairy tale based on these preferences."
        
        if characters:
            user_input += f"The story must include characters: {', '.join(characters)}. "
        if details:
            user_input += f"Include detailed descriptions: {', '.join(details)}."
            
        scenario, response = generate_scenario(system_input, user_input)
        
        # 문단 분리
        paragraphs = scenario.split("\n\n")

        # 언어별로 저장할 구조
        story_json = {"Korean": [], "English": []}

        
        k = -1

        # 언어 감지 및 분리
        for idx, para in enumerate(paragraphs):
            lang = detect(para)  
            if lang == "ko":
                story_json["Korean"].append({"scene": idx + 1, "content": para})
                k += 1
            elif lang == "en":
                story_json["English"].append({"scene": idx - k, "content": para})

        output_format = "jpeg"
        aspect_ratio = "16:9"

        # JSON 데이터를 기반으로 이미지 생성
        generated_images = generate_images_from_json(story_json['English'], output_format, aspect_ratio)

        

                


        # 결과 반환
        return render(request, '2story.html', context)


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
        return scenario, response
    
    except Exception as e:
        return f"Unexpected Error: {str(e)}"





def generate_image(description):
    url = "https://api.stable-diffusion.org/v1/generate"
    headers = {"Authorization": f"Bearer {STABLE_DIFFUSION_API_KEY}"}
    payload = {"prompt": description}
    response = requests.post(url, headers=headers, json=payload)
    return response.json().get('image_url')





# Set up Eleven Labs API
def generate_tts(text):
    url = "https://api.elevenlabs.io/v1/tts"
    headers = {
        "Authorization": f"Bearer {ELEVEN_LABS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text, "voice": "default"}
    response = requests.post(url, headers=headers, json=payload)
    file_path = f"media/tts_output_{text[:10].replace(' ', '_')}.mp3"
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return file_path


#@title Define functions

def send_generation_request(
    host,
    params,
    files = None
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
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
        "Authorization": f"Bearer {STABILITY_KEY}"
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
    response_seed = response.headers.get("seed")

    # NSFW 필터링 확인
    if finish_reason == "CONTENT_FILTERED":
        raise Warning("Generation failed due to NSFW content.")

    # 저장 경로 설정
    images_dir = os.path.join(settings.MEDIA_ROOT, 'images')
    os.makedirs(images_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)

    generated_filename = f"generated_{response_seed}.{output_format}"
    file_path = os.path.join(images_dir, generated_filename)

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(output_image)

    print(f"Image saved as: {file_path}")
    return file_path

def generate_images_from_json(json_file, output_format="jpeg", aspect_ratio="16:9"):

    # JSON 파일 읽기
    with open(json_file, "r", encoding="utf-8") as file:
        story_scenes = json.load(file)

    generated_files = []

    for scene in story_scenes:
        scene_number = scene["scene"]
        prompt = scene["content"]
        print(f"Generating image for Scene {scene_number}...")

        try:
            generated_filename = generate_image_from_prompt(
                prompt=prompt,
                negative_prompt="",
                aspect_ratio=aspect_ratio,
                seed=0,
                output_format=output_format
            )
            generated_files.append({"scene": scene_number, "file": generated_filename})
        except Warning as w:
            print(f"Scene {scene_number}: {w}")
        except Exception as e:
            print(f"Scene {scene_number}: {e}")

    print("All images generated.")
    return generated_files


