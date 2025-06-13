import os
import json
import base64
import time
import fitz
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI
from ollama import Client
from abc import ABC, abstractmethod
from tableai.readers.files import FileReader
import os

class MetadataKeysResponse(BaseModel):
    keys: List[str]

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)

class InferenceBackend(ABC):
    @abstractmethod
    async def infer(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        timeout: int
    ) -> Dict[str, Any]:
        ...


class AsyncOpenAIBackend(InferenceBackend):
    def __init__(self, api_key: str):
        # Use the correct import
        self.client = AsyncOpenAI(api_key=api_key)

    async def infer(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        timeout: int
    ) -> Dict[str, Any]:
        # Call the async-openai parse endpoint
        response = await self.client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            timeout=timeout,
            response_format={ "type": "json_object" }
        )
        try:
            response_json = json.loads(response.choices[0].message.content)
            return response_json
        except:
            return response.choices[0].message.content


class SelfHostedBackend(InferenceBackend):
    def __init__(self, host: str, username: str, password: str, request_timeout: int=180):
        # include scheme (http or https) as appropriate for your ingress
        self.host = host if host.startswith(("http://", "https://")) else f"https://{host}"
        self.username = username
        self.password = password
 
        # construct the Ollama client once, with the right header
        auth_header = self.basic_auth_header(username, password)
        self.client = Client(
            host=self.host,
            headers={"Authorization": auth_header},
        )

    def basic_auth_header(self, username: str, password: str) -> str:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return f"Basic {token}"

    def _translate_messages_for_ollama(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Take VisionInferenceClient-style messages and turn into:
        [{
          'role': 'system', 'content': <string>
        },{
          'role': 'user',
          'content': <string>,
          'images': [<base64-string>, ...]
        }]
        """
        out: List[Dict[str,Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                out.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                # content is a list of ContentItem dicts
                parts = msg["content"]
                # pull out the text piece
                text_part = next((p["text"] for p in parts if p["type"] == "text"), "")
                # pull out all base64 URLs (strip the prefix)
                images = []
                for p in parts:
                    if p["type"] == "image_url":
                        url = p["image_url"]["url"]
                        # url is "data:image/png;base64,AAA..."
                        images.append(url.split(",",1)[1])
                out.append({
                    "role": "user",
                    "content": text_part,
                    "images": images
                })
        return out

    async def infer(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        timeout: int,
        format: str = "json"
    ) -> Dict[str, Any]:
        # 1) translate
        ollama_messages = self._translate_messages_for_ollama(messages)

        # 2) call the Ollama client
        response = self.client.chat(
            model=model_name,
            messages=ollama_messages,
            format=format
        )

        # 3) response.message.content is already JSON (string or dict)
        # if it's a string, try to parse
        content = response.message.content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except ValueError:
                return {"raw": content}
        return content

    def list_models(self):
        # sanity check: list tags/models
        return self.client.list()


class VisionInferenceClient:
    def __init__(
        self,
        backend: InferenceBackend,
        default_model: str,
        default_options: Optional[Dict[str, Any]] = None
    ):
        self.backend = backend
        self.default_model = default_model
        self.default_options = default_options or {}

    def set_inference_backend(self, backend: InferenceBackend):
        self.backend = backend

    def build_message_history(self, prompt: str, page_b64: str) -> List[Dict[str, Any]]:
        base_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_b64}"}}
        ]
        return [
            {"role": "system", "content": "What text do you see in this image?"},
            {"role": "user",   "content": base_content}
        ]

    async def infer_page(
        self,
        pdf_model: Any,
        page_number: int,
        prompt: str,
        output_model: Type[OutputModelT],
        model_name: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 2,
        verbose: bool = False,
        clip_rect: Optional[List[int]] = None,
        combined: Optional[bool] = False
    ) -> Union[OutputModelT, Dict[str, Any]]:
        model = model_name or self.default_model
        page_b64 = pdf_model.get_page_base64(page_number, zoom=2.0, bounds=clip_rect, combined=combined)
        messages = self.build_message_history(prompt, page_b64)

        last_err = None
        for attempt in range(1, max_retries + 1):
            raw=None
            try:
                if verbose:
                    print(f"[Try {attempt}] sending to model={model}")
                raw = await self.backend.infer(model, messages, timeout)

                # Attempt Pydantic validation
                validated = output_model.model_validate(raw)
                return validated

            except ValidationError as ve:
                # Return raw + validation error
                if verbose:
                    print(f"[VALIDATION ERROR] {ve}")
                return {
                    "validated": None,
                    "raw": raw,
                    "validation_error": ve.errors()  # or str(ve)
                }

            except Exception as e:
                last_err = e
                if verbose:
                    print(f"[WARN] attempt {attempt} failed: {e}")
                time.sleep(0.5)

        # All attempts failed for non‐validation errors
        return {
            "validated": None,
            "raw": None,
            "error": str(last_err),
            'messages': messages
        }


    async def process_pdf(
        self,
        pdf_model: Any,
        prompt: str,
        output_model: Type[OutputModelT],
        model_name: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 2,
        verbose: bool = False,
        clip_rects: Optional[
            Union[
                Dict[int, List[List[float]]], 
                List[tuple[int, List[float]]]
            ]
        ] = None,
        combined: Optional[bool] = False
    ) -> List[Union[OutputModelT, Dict[str, Any]]]:
        """
        Process pages of a PDF, optionally with multiple clip regions per page.

        - If clip_rects is None, we do full-page inference for every page in pdf_model.
        - If clip_rects is provided, we only run inference for those page/bounds pairs.
        """
        # 1) Normalize clip_rects into a dict: page -> [rect1, rect2, ...]
        clip_map: Dict[int, List[List[float]]] = {}
        if clip_rects is not None:
            if isinstance(clip_rects, dict):
                clip_map = clip_rects
            else:
                for pg, bounds in clip_rects:
                    clip_map.setdefault(pg, []).append(bounds)

        results: List[Union[OutputModelT, Dict[str, Any]]] = []

        if clip_map:
            # Only process the specified clips
            for page_number, rects in clip_map.items():
                for bounds in rects:
                    if verbose:
                        print(f"🔪 Clipping page {page_number} to {bounds}")
                    res = await self.infer_page(
                        pdf_model=pdf_model,
                        page_number=page_number,
                        prompt=prompt,
                        output_model=output_model,
                        model_name=model_name,
                        timeout=timeout,
                        max_retries=max_retries,
                        verbose=verbose,
                        clip_rect=bounds,
                        combined=combined
                    )
                    results.append(res)
                    if verbose:
                        print(f"✅ Page {page_number} clip {bounds} done.")
        else:
            # No clips—process every page full size
            for page_number, _ in pdf_model:
                res = await self.infer_page(
                    pdf_model=pdf_model,
                    page_number=page_number,
                    prompt=prompt,
                    output_model=output_model,
                    model_name=model_name,
                    timeout=timeout,
                    max_retries=max_retries,
                    verbose=verbose,
                    clip_rect=None
                )
                results.append(res)
                if verbose:
                    print(f"✅ Page {page_number} full-page done.")

        return results