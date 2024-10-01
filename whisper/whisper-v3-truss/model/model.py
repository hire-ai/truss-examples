from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

import requests
import torch

import whisper
import time


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def download_file_with_retries(
        self, url: str, max_retries: int = 3, wait_seconds: int = 2
    ) -> bytes:
        url = url.strip()
        attempt = 0
        while attempt < max_retries:
            try:
                print(
                    f"Attempting to download file from >>{url}<<, attempt {attempt + 1}"
                )

                resp = requests.get(url)
                resp.raise_for_status()  # Raise an exception for HTTP errors
                print(
                    f"File downloaded successfully from >>{url}<< on attempt {attempt + 1}"
                )
                return resp.content
            except requests.exceptions.RequestException as e:
                print(f"Error downloading file from {url}: {e}")
                attempt += 1
                if attempt < max_retries:
                    print(f"Retrying in {wait_seconds} seconds...")
                    time.sleep(wait_seconds)
                else:
                    print(
                        f"Failed to download the file from {url} after {max_retries} attempts."
                    )
                    raise
        return b""

    def preprocess(self, request: Dict) -> Dict:
        print(f"Received URL: >>{repr(request['url'])}<<")

        try:
            content = self.download_file_with_retries(request["url"])
        except requests.exceptions.RequestException as e:
            print(f"Failed to download the file from {request['url']}: {e}")
            raise
        return {"response": content}

    def load(self):
        self.model = whisper.load_model(
            Path(str(self._data_dir)) / "weights" / "large-v3.pt", self.device
        )

    def predict(self, request: Dict) -> Dict:
        print("Starting prediction...")
        with NamedTemporaryFile() as fp:
            fp.write(request["response"])
            print("Temporary file created: ", fp.name)

            result = whisper.transcribe(
                self.model,
                fp.name,
                temperature=0,
                compression_ratio_threshold=1.35,
                beam_size=5,
                best_of=5,
                logprob_threshold=2.8,
                # no_speech_threshold=0.4
                word_timestamps=True,
            )
            print("Transcription completed successfully")

            segments = [
                {"start": r["start"], "end": r["end"], "text": r["text"]}
                for r in result["segments"]
            ]
        return {
            "language": whisper.tokenizer.LANGUAGES[result["language"]],
            "segments": segments,
            "text": result["text"],
        }
