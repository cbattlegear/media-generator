from openai import OpenAI
import os
import json
import requests
from io import BytesIO


# Parent class for local OpenAI-compatible API models
class localOpenAIModel:

    def __init__(self):
        self.base_url = ""
        self.key = ""
        self.model = ""
        self.client = None
        self.system_prompt = ""
        self.user_prompt = ""

    def to_json(self):
        return {
            "base_url": self.base_url,
            "model": self.model,
        }


# Child class for local OpenAI-compatible Text model
class localOpenAIText(localOpenAIModel):
    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("LOCAL_OPENAI_ENDPOINT")
        self.key = os.getenv("LOCAL_OPENAI_API_KEY", "no-key-required")
        self.model = os.getenv("LOCAL_OPENAI_TEXT_MODEL")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.key,
        )

    def generateResponse(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
            ],
            max_completion_tokens=2000,
        )

        return response.choices[0].message.content


# Child class for local OpenAI-compatible Image model
class localOpenAIImage(localOpenAIModel):
    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("LOCAL_OPENAI_IMAGE_ENDPOINT", os.getenv("LOCAL_OPENAI_ENDPOINT"))
        self.key = os.getenv("LOCAL_OPENAI_API_KEY", "no-key-required")
        self.model = os.getenv("LOCAL_OPENAI_IMAGE_MODEL")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.key,
        )

    def generateImage(self):
        result = self.client.images.generate(
            model=self.model,
            prompt=self.user_prompt,
            n=1,
            size="1024x1792",
        )

        json_response = json.loads(result.model_dump_json())

        image_url = json_response["data"][0]["url"]
        return BytesIO(requests.get(image_url).content)


# Child class for local OpenAI-compatible Vision model
class localOpenAIVision(localOpenAIModel):
    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("LOCAL_OPENAI_VISION_ENDPOINT", os.getenv("LOCAL_OPENAI_ENDPOINT"))
        self.key = os.getenv("LOCAL_OPENAI_API_KEY", "no-key-required")
        self.model = os.getenv("LOCAL_OPENAI_VISION_MODEL", os.getenv("LOCAL_OPENAI_TEXT_MODEL"))
        self.image_base64 = ""
        self.mime_type = "image/png"

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.key,
        )

    def generateResponse(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{self.mime_type};base64,{self.image_base64}"
                            },
                        },
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ],
            max_completion_tokens=2000,
        )
        return response.choices[0].message.content
