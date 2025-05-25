import requests, time, base64, json
from typing import Optional, Dict, Any, Type, List, TypeVar, Generic, Union
import fitz
from pydantic.generics import GenericModel
from pydantic import BaseModel, Field, Extra, ValidationError

### Aplication imports ###
from tableai.nodes.pdf_node import PDFModel
from api.models.requests import (
    PdfVisionModelRequest,
    VisionModelOptions
)
### ------------------ ###

PdfModelType = TypeVar("PdfModelType", bound=BaseModel)
OutputModelType = TypeVar("OutputModelType", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)

class VisionInferenceClient(Generic[OutputModelT]):
    def __init__(
        self,
        model_library: Optional[Dict[str, tuple]] = None,
        model_choice: str = 'mistralVL',
        username: Optional[str] = None,
        password: Optional[str] = None,
        default_options: Optional[Dict[str, Any]] = None,
        default_format: str = "json",
        stream: bool = False,
        page_limit: Optional[int] = None
    ):
        self.model_library = model_library or {
            'llava7b': ('llava:7b', 'ollama-embeddings-ollama.portal-ai.tools'),
            'gemma3': ('gemma3:27b', 'ollama-embeddings-ollama.portal-ai.tools'),
            'mistralVL': ('mistral-small3.1:24b-max-layers', 'ollama-medium-ollama.portal-ai.tools'),
            'llamaVL': ('llama3.2-vision', 'ollama-embeddings-ollama.portal-ai.tools')
        }
        self.model_choice = model_choice
        self.username = username
        self.password = password
        self.default_options = default_options or {"temperature": 0, "top_k": 1, "top_p": 1}
        self.default_format = default_format
        self.stream = stream
        self.page_limit = page_limit

    @staticmethod
    def convert_pdf_page_to_base64(page, zoom=2.0):
        """Render a PDF page to base64 PNG."""
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

    def get_model_and_endpoint(self, model_choice: Optional[str] = None):
        choice = model_choice or self.model_choice
        if choice not in self.model_library:
            raise ValueError(f"Model '{choice}' not found in model_library.")
        return self.model_library[choice]

    def infer_page(
        self,
        request: PdfVisionModelRequest,
        pdf_model: 'PDFModel',
        page_number: int,
        output_model: Type[OutputModelT],
        username: Optional[str] = None,
        password: Optional[str] = None,
        format: Optional[str] = None,
        stream: Optional[bool] = None,
        verbose: bool = False
    ) -> Optional[OutputModelT | dict]:
        model_name, model_endpoint = self.get_model_and_endpoint(request.model_choice)
        user = username or self.username
        pw = password or self.password

        image_b64 = pdf_model.get_page_base64(page_number, zoom=request.zoom or 2.0)

        payload = {
            "model": model_name,
            "prompt": request.prompt,
            "stream": stream if stream is not None else self.stream,
            "format": format or self.default_format,
            "images": [image_b64],
            "options": request.options.model_dump() if request.options else self.default_options,
        }
        url = f'https://{model_endpoint}/api/generate'

        last_error = None
        for attempt in range(1, (request.max_attempts or 2) + 1):
            try:
                if verbose:
                    print(f"Attempt {attempt}: POST {url} | Model: {model_name}")
                resp = requests.post(url, json=payload, auth=(user, pw) if user and pw else None, timeout=request.timeout or 166)
                resp.raise_for_status()
                resp_data = resp.json()
                raw_str = resp_data.get("response", "")
                try:
                    response_obj = json.loads(raw_str)
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"[ERROR] JSON decoding failed: {e}")
                    return {
                        "validated": None,
                        "raw": raw_str,
                        "error": f"JSON decode error: {str(e)}"
                    }
                try:
                    validated = output_model.model_validate(response_obj)
                except ValidationError as e:
                    if verbose:
                        print(f"[ERROR] LLM output failed schema validation:\n{e}")
                    return {
                        "validated": None,
                        "raw": raw_str,
                        "json": response_obj,
                        "validation_error": str(e),
                        "page_number": page_number
                    }
                return {'result': validated, 'page_number': page_number}
            except (requests.RequestException, KeyError) as e:
                last_error = e
                if verbose:
                    print(f"[WARN] Inference failed (attempt {attempt}): {e}")
                time.sleep(0.5)
        if verbose:
            print(f"[ERROR] All attempts failed for inference.")
        return {
            "validated": None,
            "raw": None,
            "error": str(last_error),
            "page_number": page_number
        }

    def build_schema_guided_prompt(self, prompt: str, output_model: type[BaseModel]) -> str:
        """
        Combines the user prompt and the output model schema as a single prompt string.
        """
        schema_json = output_model.schema_json(indent=2)
        return (
            f"{prompt}\n\n"
            f"Return your answer as a valid JSON object with this schema:\n"
            f"{schema_json}"
        )

    def process_pdf(
        self,
        request: PdfVisionModelRequest,
        pdf_model: 'PDFModel',
        output_model: Type[OutputModelT],
        username: Optional[str] = None,
        password: Optional[str] = None,
        format: Optional[str] = None,
        stream: Optional[bool] = None,
        verbose: bool = True
    ) -> List[Any]:
        results = []
        schema_prompt = self.build_schema_guided_prompt(request.prompt, output_model)
        print(f"schema_prompt: {schema_prompt}")
        print(f"Processing {pdf_model.effective_limit} pages from {pdf_model.path}")

        for page_number, page in pdf_model:
            result = self.infer_page(
                request=request,
                pdf_model=pdf_model,
                page_number=page_number,
                output_model=output_model,
                username=username,
                password=password,
                format=format,
                stream=stream,
                verbose=verbose,
            )
            results.append(result)
            print(f"Page {page_number+1} processed.")
        print("Done.")
        return results