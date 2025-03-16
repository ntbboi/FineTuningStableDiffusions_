import base64
import json
from pathlib import Path
from openai import OpenAI

# OpenAI API 키 설정
client = OpenAI(api_key='apikey')

# 이미지 폴더 경로 및 출력 파일 경로
image_folder = Path(r"D:\CV\finetuning\preprocessed_512_front\merged_folder")
output_jsonl = r"D:\CV\finetuning\preprocessed_512_front\metadata2.jsonl"

# 이미지 설명 생성 함수 (GPT-4o 모델 사용)
def generate_caption(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate a concise image description that includes fashion style, colors, and details optimized for Stable Diffusion fine-tuning."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=50
        )

        # 응답이 없는 경우 처리
        if not response.choices or not response.choices[0].message.content:
            print(f"⚠️ [Warning] No response for: {image_path.name}")
            return "No description available."

        caption = response.choices[0].message.content.strip()
        return caption

    except Exception as e:
        print(f"❌ [Error] Failed to process {image_path.name}: {e}")
        return "Error in generating description."

# JSONL 파일 생성
with open(output_jsonl, 'w', encoding='utf-8') as out_file:
    for img_file in sorted(image_folder.iterdir()):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            print(f"Processing: {img_file.name}")
            caption = generate_caption(img_file)
            json_line = json.dumps({"file_name": img_file.name, "text": caption}, ensure_ascii=False)
            out_file.write(json_line + "\n")

print("✅ JSONL file created successfully.")
