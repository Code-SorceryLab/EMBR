"""The language-model runner: step 5 of the pipeline.

EMBR's contribution is the memory layer, not the model, so the model sits behind a tiny
interface and can be swapped freely. Three runners share it:

  * `StubRunner` - deterministic echo, no weights, no network. The default, so the whole
    pipeline (logging, state update, scoring, retrieval, prompt building) runs on any
    machine and every test stays fast and hermetic.
  * `OllamaRunner` - a real local (or hosted) model over Ollama's HTTP API, standard
    library only, so the core gains no dependency. This is the conventional-model arm of
    the bake-off.
  * `OuroRunner` - the thesis model, Ouro 1.4B, loaded in-process through transformers.

`GenerationSettings` is the one place sampling is configured, which is what lets the
bake-off hold temperature, top-p, length, and seed equal across every model.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Defaults kept as named constants because the config layer and the tests both need them.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OURO_MODEL = "ByteDance/Ouro-1.4B"
DEFAULT_ENV_FILE = ".env"
OLLAMA_API_KEY_NAME = "OLLAMA_API_KEY"


@runtime_checkable
class ModelRunner(Protocol):
    """Anything that can turn a prompt into a reply. One method, on purpose."""

    def generate(self, prompt: str) -> str: ...


class ModelUnavailableError(RuntimeError):
    """A runner could not produce a reply: no daemon, no weights, or a bad response.

    Its own exception type so callers can distinguish "the model is not set up here" from
    a genuine bug, and so a failed eval run never passes silently as an empty reply.
    """


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling knobs, in one immutable place.

    Frozen so a single shared instance can safely be the default argument of every runner,
    and so a bake-off can pass the exact same object to each model under comparison.
    """

    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 120
    seed: int = 7


# The shared default instance. Immutable, so sharing it is safe.
DEFAULT_GENERATION_SETTINGS = GenerationSettings()


class StubRunner:
    """Deterministic stand-in model: no weights, no network, no GPU.

    It does not actually reason; it returns a short, obviously-fake line so the surrounding
    pipeline (logging, state update, scoring, retrieval, prompt building) can be exercised
    and demonstrated before the real model is wired in.
    """

    def __init__(self, label: str = "stub") -> None:
        self.label = label

    def generate(self, prompt: str) -> str:
        # Echo just the player's line back so a demo turn visibly responds to input,
        # while staying clearly marked as a placeholder reply.
        player_line = ""
        for line in prompt.splitlines():
            if line.startswith("The player says:"):
                player_line = line.split(":", 1)[1].strip().strip('"')
                break
        return f"[{self.label} reply] I heard you say: {player_line!r}"


def read_ollama_api_key(
    env_file: str | Path = DEFAULT_ENV_FILE, variable: str = OLLAMA_API_KEY_NAME
) -> str | None:
    """Return the Ollama API key from the environment, else from a local .env, else None.

    Only the hosted endpoint needs a key, so "absent" is a normal, non-exceptional answer:
    the local daemon works fine without one. The key is never logged or echoed anywhere,
    and a blank value counts as absent.
    """
    from_environment = os.environ.get(variable, "").strip()
    if from_environment:
        return from_environment

    source = Path(env_file)
    try:
        raw = source.read_bytes()
    except OSError:
        return None  # no .env here (or unreadable): absent, not an error

    # Windows shells do not write UTF-8 by default: PowerShell's `>` emits UTF-16LE with a
    # BOM and Set-Content uses the ANSI codepage, so a hand-made .env is routinely not UTF-8.
    # Decoding strictly raises inside build_model and takes down the entire model path, which
    # is a hostile failure for a file that is optional in the first place. Detect the encoding
    # from the byte-order mark, fall back to a NUL scan for BOM-less UTF-16, and treat bytes
    # that decode as nothing the same way as a missing file.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ("utf-16",)
    elif b"\x00" in raw:
        candidates = ("utf-16-le", "utf-16-be")
    else:
        candidates = ("utf-8-sig",)

    for encoding in candidates:
        try:
            lines = raw.decode(encoding).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        return None

    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        name, _, raw_value = entry.partition("=")
        if name.strip() != variable:
            continue
        value = raw_value.strip().strip("\"'").strip()
        return value or None
    return None


def _require_non_empty_reply(text: str, remedy: str) -> str:
    """Return `text` stripped, or fail loudly if the model produced nothing.

    A blank reply is never useful: it would flow into the pipeline and silently flatten a
    tone measurement, so both real runners route their output through here and every empty
    completion becomes an error that names the knob to change.
    """
    reply = text.strip()
    if reply:
        return reply
    raise ModelUnavailableError(f"The model returned an empty reply. {remedy}")


class OllamaRunner:
    """A real model served by Ollama, over its HTTP API, using the standard library only.

    The same class serves the local daemon and the hosted endpoint: pass
    `host="https://ollama.com"` plus an `api_key` and the request carries a bearer token,
    otherwise no auth header is sent at all. Nothing here ever logs the key.
    """

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_OLLAMA_HOST,
        api_key: str | None = None,
        settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")  # so a trailing slash cannot double up in the URL
        self.api_key = api_key
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    @property
    def label(self) -> str:
        """Which model actually served the run, for a run directory to record.

        Cloud and local are the same class differing only by a bearer token, so the host
        is part of the name: two runs whose only difference is where the model ran must
        not be recorded under one label.
        """
        where = "cloud" if self.api_key else "local"
        return f"{self.model} ({where})"

    def __repr__(self) -> str:
        # Explicitly reports only *whether* a key is set, so a traceback or log line that
        # prints a runner can never expose the secret itself.
        return (
            f"OllamaRunner(model={self.model!r}, host={self.host!r}, "
            f"api_key={'set' if self.api_key else 'none'})"
        )

    def _build_request(self, prompt: str) -> urllib.request.Request:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "num_predict": self.settings.max_new_tokens,  # Ollama's name for max tokens
                "seed": self.settings.seed,
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def generate(self, prompt: str) -> str:
        request = self._build_request(prompt)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise ModelUnavailableError(self._http_error_hint(error)) from error
        except urllib.error.URLError as error:
            raise ModelUnavailableError(
                f"Could not reach the Ollama daemon at {self.host} ({error.reason}). "
                f"Start it with `ollama serve`, or point host= at a running one."
            ) from error
        except json.JSONDecodeError as error:
            raise ModelUnavailableError(
                f"Ollama at {self.host} returned a body that is not JSON."
            ) from error

        if "response" not in body:
            raise ModelUnavailableError(
                f"Ollama at {self.host} returned no 'response' field for model "
                f"{self.model!r}; got keys {sorted(body)}."
            )
        return _require_non_empty_reply(str(body["response"]), self._empty_reply_remedy(body))

    def _empty_reply_remedy(self, body: dict[str, Any]) -> str:
        """Explain a blank completion using the metadata Ollama returns alongside it."""
        # A reasoning model streams its chain of thought into "thinking" and can exhaust
        # num_predict there, ending with done_reason "length" and an empty "response".
        spent_on_thinking = (
            " It spent the budget on its hidden 'thinking' channel instead."
            if str(body.get("thinking") or "").strip()
            else ""
        )
        return (
            f"Ollama at {self.host} gave model {self.model!r} "
            f"{self.settings.max_new_tokens} tokens and got nothing back "
            f"(done_reason={body.get('done_reason')!r}).{spent_on_thinking} "
            f"Raise GenerationSettings.max_new_tokens, or choose a non-reasoning model."
        )

    def _http_error_hint(self, error: urllib.error.HTTPError) -> str:
        """Turn an HTTP status into an instruction the reader can act on."""
        if error.code == 404:
            return (
                f"Ollama at {self.host} does not have model {self.model!r}. "
                f"Pull it first: `ollama pull {self.model}`."
            )
        if error.code in (401, 403):
            return (
                f"Ollama at {self.host} rejected the credentials for model {self.model!r}. "
                f"Set {OLLAMA_API_KEY_NAME} for a hosted host, or drop the key for a local one."
            )
        return (
            f"Ollama at {self.host} failed with HTTP {error.code} ({error.reason}) for model "
            f"{self.model!r}."
        )


def detect_torch_device() -> str:
    """Pick the fastest device torch can see: cuda, else mps (Apple), else cpu."""
    import torch  # lazy: importing embr must not pull torch in

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)  # absent on older torch builds
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


# Some chat-tuned checkpoints open a completion with a role label. Anchored, and with a word
# boundary, so a real reply that merely starts with the word "Assistants" survives untouched.
_ASSISTANT_ARTEFACT = re.compile(r"^\s*assistant\b\s*:?\s*", re.IGNORECASE)


def strip_assistant_prefix(text: str) -> str:
    """Drop a leading "Assistant"/"Assistant:" role label the model may emit, and trim."""
    return _ASSISTANT_ARTEFACT.sub("", text, count=1).strip()


class OuroRunner:
    """The thesis model: ByteDance Ouro 1.4B, loaded in-process through transformers.

    Ouro is a *looped* model: instead of stacking more layers it repeats the same internal
    computation several times per token, which is why it is slower per token than a
    conventional model of similar size and why it is the interesting arm of the bake-off.

    Practical notes for whoever runs this next:

      * It needs **transformers 4.x** (5.x breaks its remote code) and
        `trust_remote_code=True`.
      * Weights load in float16 on cuda, else mps, else cpu, and **loading costs about
        10 seconds**, so the model is loaded lazily on the first `generate` and then cached
        on the instance; later calls reuse it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OURO_MODEL,
        device: str | None = None,
        settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
    ) -> None:
        self.model_name = model_name
        self.device = device  # None until the first generate auto-detects it
        self.settings = settings
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def label(self) -> str:
        """Which model served the run. Carries the device, since latency depends on it, and
        resolves it eagerly so the label is the same before and after the weights load."""
        if self.device is None:
            self.device = detect_torch_device()
        return f"{self.model_name} ({self.device})"

    @property
    def is_loaded(self) -> bool:
        """Whether the weights are in memory yet (useful for a UI or a timing log)."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Lazy imports: importing embr stays light for every caller that uses the stub.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = detect_torch_device()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, dtype=torch.float16
            )
        except Exception as error:  # noqa: BLE001 - re-raised with an actionable hint
            raise ModelUnavailableError(
                f"Could not load {self.model_name!r}. It needs transformers 4.x (5.x breaks "
                f"its remote code), trust_remote_code=True, and the weights in the Hugging "
                f"Face cache. Underlying error: {error}"
            ) from error
        self._model = model.to(self.device).eval()

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        import torch

        torch.manual_seed(self.settings.seed)  # same knob as Ollama's seed option
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            produced = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                do_sample=self.settings.temperature > 0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only what the model added: `generate` returns prompt + completion, and
        # returning the prompt back to the pipeline would corrupt every tone measurement.
        new_tokens = produced[0][prompt_token_count:]
        completion = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _require_non_empty_reply(
            strip_assistant_prefix(completion),
            f"{self.model_name} added no usable tokens on {self.device}; raise "
            f"GenerationSettings.max_new_tokens (now {self.settings.max_new_tokens}).",
        )
