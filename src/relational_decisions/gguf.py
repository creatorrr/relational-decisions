"""Pinned local llama.cpp decision backends, with owned server processes."""

import atexit
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .decisions import Candidate
from .h2o import DEFAULT_MODEL, DEFAULT_REVISION, H2OLightningBackend, PROMPT
from .prompts import WORLD_PREFIX, build_tasks, decode_scores

BUILD = "b11515-3d65c90d0"
MODELS = {
    "h2o-q8": {
        "repo": "mradermacher/h2o-lightning-4b-GGUF",
        "revision": "7a62fde885ed87e7b22a2195099c6b204381138f",
        "filename": "h2o-lightning-4b.Q8_0.gguf",
        "sha256": "3428e91d1a023a6f0ee953a63f6e5c741df6148901d5287aa73817b87abe0a9d",
    },
    "d1-q8": {
        "repo": "LiquidAI/d1-3B-GGUF",
        "revision": "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563",
        "filename": "d1-3B-Q8_0.gguf",
        "sha256": "2f0942d5a5f64cf69c3356d2be439b71644b1f0eb3a580912a5a0179eeaba77d",
    },
}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def label_probabilities(response, ids, temperature):
    """Full-vocabulary log normalizer cancels in the label-only softmax."""
    if response.get("truncated") is not False:
        raise ValueError("Missing truncation status or truncated prompt")
    rows = response["completion_probabilities"][0]["top_logprobs"]
    scores = {row["id"]: row["logprob"] for row in rows}
    if any(i not in scores or not math.isfinite(scores[i]) for i in ids):
        raise ValueError("Answer token absent or invalid in returned log probabilities")
    maximum = max(scores[i] for i in ids)
    weights = [math.exp((scores[i] - maximum) / temperature) for i in ids]
    return [w / sum(weights) for w in weights]


def common_prefix(sequences):
    result = []
    for values in zip(*sequences):
        if len(set(values)) != 1:
            break
        result.append(values[0])
    return result


class GGUFBackend:
    def __init__(self, model, *, server_binary=None, port=8767, threads=4):
        from huggingface_hub import hf_hub_download

        self.kind = model
        spec = MODELS[model]
        path = Path(
            hf_hub_download(spec["repo"], spec["filename"], revision=spec["revision"])
        )
        if file_sha(path) != spec["sha256"]:
            raise ValueError("GGUF checksum mismatch")
        binary = Path(server_binary or os.environ["LLAMA_SERVER_BIN"]).resolve()
        self.url = f"http://127.0.0.1:{port}"
        self.process = None
        self.log = tempfile.TemporaryFile()
        self.server_peak_rss_kib = 0
        self.http_seconds = 0.0
        self.http_calls = 0
        flags = [
            "-t",
            str(threads),
            "-tb",
            str(threads),
            "-b",
            "512",
            "--cache-ram",
            "0",
            "--no-context-shift",
        ]
        if model == "h2o-q8":
            flags += [
                "-c",
                "2048",
                "-np",
                "1",
                "-ub",
                "128",
                "--ctx-checkpoints",
                "32",
                "--checkpoint-min-step",
                "0",
            ]
        else:
            flags += ["-c", "16384", "-np", "8", "-ub", "512"]
        # Refuse to accidentally attach to an existing service on this port.
        try:
            urllib.request.urlopen(self.url + "/health", timeout=1).close()
        except urllib.error.URLError:
            pass
        else:
            raise ValueError("Server port already in use")
        self.process = subprocess.Popen(
            [
                str(binary),
                "-m",
                str(path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                *flags,
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.close)
        try:
            deadline = time.monotonic() + 120
            while True:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    raise RuntimeError(self.log.read().decode(errors="replace")[-4000:])
                try:
                    props = self.request("props")
                    break
                except urllib.error.URLError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("llama-server did not start")
                    time.sleep(0.2)
            if props["build_info"] != BUILD or Path(props["model_path"]) != path:
                raise ValueError("Unexpected server build or loaded model")
            template = Candidate(
                "candidate", ("p",), ("n",), "{positive}", "{negative}"
            )
            self._identity = {
                "backend": "llama.cpp-" + model,
                "adapter_version": "native-choice-gguf-v1",
                "model": spec,
                "llama_build": BUILD,
                "server_binary_sha256": file_sha(binary),
                "runtime_libraries_sha256": {
                    p.name: file_sha(p) for p in sorted(binary.parent.glob("*.so"))
                },
                "server_flags": flags,
                "prompt_version": PROMPT,
                "world_prefix": WORLD_PREFIX,
                "task_templates": [asdict(t) for t in build_tasks([template], PROMPT)],
                "score_composition": "independent-positive-negative-product",
                "attention": "causal; shared prefix with independent question suffixes",
                "truncation": "reject",
            }
            if model == "h2o-q8":
                self._prepare_h2o()
            else:
                self._identity.update(
                    endpoint="/v1/systemone",
                    question_group_size_max=8,
                    readout="max over A/space-A and B/space-B token aliases, then softmax",
                    temperature=1.0,
                    calibration="GGUF has no temperature metadata; llama.cpp default 1.0",
                    prefix_reuse="native common-prefix parent with causal child branches",
                )
        except BaseException:
            self.close()
            raise

    def _prepare_h2o(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        path = Path(
            snapshot_download(
                DEFAULT_MODEL,
                revision=DEFAULT_REVISION,
                allow_patterns=[
                    "tokenizer*.json",
                    "chat_template.jinja",
                    "config.json",
                    "h2o_lightning_shim.py",
                    "serve_config.json",
                ],
            )
        )
        spec = importlib.util.spec_from_file_location(
            "_h2o_gguf_contract", path / "h2o_lightning_shim.py"
        )
        shim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shim)
        self.renderer = H2OLightningBackend.__new__(H2OLightningBackend)
        self.renderer.tokenizer = AutoTokenizer.from_pretrained(path)
        self.renderer.contract = shim.Contract(
            json.loads((path / "serve_config.json").read_text()), env={}
        )
        self._identity.update(
            endpoint="/completion",
            temperature=0.8,
            calibration="published H2O choice temperature; no benchmark calibration",
            readout="verified leading-space A/B tokens; pre-sampling logprobs; label softmax",
            prefix_reuse="reset and prefill exact group common prefix, then cached suffixes",
            n_probs=128,
            prompt_model=DEFAULT_MODEL,
            prompt_revision=DEFAULT_REVISION,
            prompt_source_sha256={
                p.name: file_sha(p)
                for p in path.iterdir()
                if p.name
                in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "h2o_lightning_shim.py",
                    "serve_config.json",
                )
            },
        )

    @property
    def identity(self):
        return self._identity

    def request(self, route, payload=None):
        tick = time.perf_counter()
        req = urllib.request.Request(
            self.url + "/" + route,
            None if payload is None else json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        self.http_seconds += time.perf_counter() - tick
        self.http_calls += 1
        return result

    def assess(self, world, candidates):
        if not candidates:
            return {}
        if self.kind == "d1-q8":
            tasks = build_tasks(candidates, PROMPT)
            scores = {}
            for start in range(0, len(tasks), 8):
                group = tasks[start : start + 8]
                questions = {
                    t.name: {
                        "type": "choice",
                        "instructions": t.instruction,
                        "criteria": t.labels,
                    }
                    for t in group
                }
                result = self.request(
                    "v1/systemone",
                    {"state": WORLD_PREFIX + world, "questions": questions},
                )
                if (
                    set(result["answers"]) != set(questions)
                    or result["usage"]["output_tokens"] != 0
                ):
                    raise ValueError("Invalid native decision response")
                for task in group:
                    values = result["answers"][task.name]["probabilities"]
                    if (
                        set(values) != set(task.labels)
                        or any(
                            not math.isfinite(v) or not 0 <= v <= 1
                            for v in values.values()
                        )
                        or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6)
                    ):
                        raise ValueError("Invalid decision probabilities")
                    scores[task.name] = values
        else:
            tasks, requests = self.renderer.requests(world, candidates)
            for text, tokens, _ in requests:
                if len(tokens) + 1 > 2048:
                    raise ValueError("Prompt exceeds frozen GGUF context")
                actual = self.request(
                    "tokenize",
                    {"content": text, "add_special": False, "parse_special": True},
                )["tokens"]
                if actual != tokens:
                    raise ValueError(
                        "GGUF tokenizer differs from pinned native tokenizer"
                    )
            prefix = common_prefix([r[1] for r in requests])
            # Reset at every group so results do not depend on prior worlds.
            self.request(
                "completion",
                {
                    "prompt": prefix,
                    "n_predict": 0,
                    "cache_prompt": False,
                    "temperature": -1,
                    "samplers": [],
                },
            )
            scores = {}
            for task, (_, tokens, ids) in zip(tasks, requests):
                response = self.request(
                    "completion",
                    {
                        "prompt": tokens,
                        "n_predict": 1,
                        "temperature": -1,
                        "n_probs": 128,
                        "post_sampling_probs": False,
                        "cache_prompt": True,
                        "seed": 0,
                        "samplers": [],
                    },
                )
                if response["tokens_evaluated"] != len(tokens):
                    raise ValueError("Server did not evaluate the complete prompt")
                scores[task.name] = dict(
                    zip(task.labels, label_probabilities(response, ids, 0.8))
                )
        return decode_scores(tasks, scores)

    def resources(self):
        if self.process is not None and self.process.poll() is None:
            for line in (
                Path(f"/proc/{self.process.pid}/status").read_text().splitlines()
            ):
                if line.startswith("VmHWM:"):
                    self.server_peak_rss_kib = max(
                        self.server_peak_rss_kib, int(line.split()[1])
                    )
        return {
            "server_peak_rss_kib": self.server_peak_rss_kib,
            "http_seconds": self.http_seconds,
            "http_calls": self.http_calls,
        }

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.resources()
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
