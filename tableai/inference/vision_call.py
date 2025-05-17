import requests
import base64
import json
import fitz  # PyMuPDF
import time
from typing import Optional, Dict, Any, List

class VisionInferenceClient:
    def __init__(
        self,
        model_library: Optional[Dict[str, tuple]] = None,
        model_choice: str = 'mistralVL',
        username: Optional[str] = None,
        password: Optional[str] = None,
        default_options: Optional[Dict[str, Any]] = None,
        default_format: str = "json",
        stream: bool = False,
    ):
        """
        Initialize the VisionInferenceClient.
        
        Args:
            model_library (dict): Maps model_choice to (model_name, endpoint).
            model_choice (str): Default model choice key.
            username (str): Username for basic auth.
            password (str): Password for basic auth.
            default_options (dict): Default options for inference.
            default_format (str): Default output format.
            stream (bool): Stream output or not.
        """
        self.model_library = model_library or {
            'mistralVL': ('mistral-small3.1:24b', 'ollama-medium-ollama.portal-ai.tools'),
            'llamaVL':   ('llama3.2-vision', 'ollama-embeddings-ollama.portal-ai.tools')
        }
        self.model_choice = model_choice
        self.username = username
        self.password = password
        self.default_options = default_options or {"temperature": 0, "top_k": 1, "top_p": 1}
        self.default_format = default_format
        self.stream = stream

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
        prompt: str,
        page: Any,
        model_choice: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        zoom: float = 2.0,
        format: Optional[str] = None,
        stream: Optional[bool] = None,
        verbose: bool = False,
    ) -> dict:
        """Run vision inference on a single PDF page."""
        model_name, model_endpoint = self.get_model_and_endpoint(model_choice)
        user = username or self.username
        pw = password or self.password

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream if stream is not None else self.stream,
            "format": format or self.default_format,
            "images": [self.convert_pdf_page_to_base64(page, zoom=zoom)],
            "options": options or self.default_options
        }
        url = f'https://{model_endpoint}/api/generate'
        if verbose:
            print(f"POST {url} | Model: {model_name}")
        resp = requests.post(url, json=payload, auth=(user, pw) if user and pw else None)
        resp.raise_for_status()
        resp_data = resp.json()
        # The actual response format might differ depending on model endpoint
        return json.loads(resp_data['response'])

    def process_pdf(
        self,
        pdf_path: str,
        prompt: str,
        model_choice: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        zoom: float = 2.0,
        format: Optional[str] = None,
        stream: Optional[bool] = None,
        verbose: bool = True,
    ) -> (List[dict], List[dict]):
        """
        Process all pages in a PDF using vision inference and track timing.
        
        Returns: (page_results, timing_data)
        """
        doc = fitz.open(pdf_path)
        results = []
        timings = []

        if verbose:
            print(f"Processing {len(doc)} pages from {pdf_path}")

        for i, page in enumerate(doc):
            start_time = time.time()
            result = self.infer_page(
                prompt=prompt,
                page=page,
                model_choice=model_choice,
                username=username,
                password=password,
                options=options,
                zoom=zoom,
                format=format,
                stream=stream,
                verbose=verbose,
            )
            elapsed = time.time() - start_time
            timings.append({"page": i, "time_seconds": elapsed})
            results.append(result)
            if verbose:
                print(f"Page {i+1}/{len(doc)} processed in {elapsed:.2f} seconds")
        
        # Print summary
        if verbose and timings:
            total = sum(x["time_seconds"] for x in timings)
            avg = total / len(timings)
            slowest = max(timings, key=lambda x: x["time_seconds"])
            fastest = min(timings, key=lambda x: x["time_seconds"])
            print(f"\nSummary:")
            print(f"Total processing time: {total:.2f} seconds")
            print(f"Average time per page: {avg:.2f} seconds")
            print(f"Fastest page: {fastest['page']+1} ({fastest['time_seconds']:.2f} s)")
            print(f"Slowest page: {slowest['page']+1} ({slowest['time_seconds']:.2f} s)")

        return results, timings