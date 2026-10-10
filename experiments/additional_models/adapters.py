"""Pinned native decision adapters for the additional-model diagnostic."""

import importlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from relational_decisions.gguf import GGUFBackend, file_sha
from relational_decisions.prompts import WORLD_PREFIX, build_tasks, decode_scores

K2_REVISION = "0648c43e30d20d85b621e3604c397f8af4d5b1de"
LUX_REVISION = "8576ef7a1c13b2b33b414eda1cfc8d81f816dc68"
LUX_SHA = "6f9150bad413c4627ef457e1225b757d3018ce0319f8d80b6cbf92efbbcd5590"
RUNTIME_COMMIT = "c9f5974870fc7982ab2f424bc0840af256189719"
PROMPT = "binary-reports-v2"


class K2Backend:
    def __init__(self, path, threads=4):
        import torch
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        from importlib.metadata import version

        path = Path(path).resolve()
        # Publisher jev package is reviewed and pinned with the model snapshot.
        sys.path.insert(0, str(path))
        api = importlib.import_module("jev.model")
        self.encoding = importlib.import_module("jev.encode")
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = api.DecisionModel(str(path), dtype=torch.bfloat16, attn="sdpa")
        self.model.head.load_state_dict(load_file(path / "pointer_head.safetensors"))
        cfg = json.loads((path / "decision_config.json").read_text())
        self.model.temperature.fill_(cfg["temperature"])
        self.model.eval()
        self.encoder = self.encoding.Encoder(self.tokenizer, 8192, 7168)
        self.calls = self.tokens = 0
        self.forward_seconds = 0.0
        self.identity = {
            "backend": "k2-native-cpu", "model": "IFM/K2-Type-0.9B",
            "revision": K2_REVISION, "prompt_version": PROMPT,
            "world_prefix": WORLD_PREFIX, "dtype": "bfloat16 backbone / float32 pointer head",
            "device": "cpu", "threads": threads, "temperature": cfg["temperature"],
            "readout": "publisher pointer head; published temperature; native softmax without HTTP rounding",
            "attention": "publisher block-causal mask; per-question position reset",
            "option_order": ["yes", "no"],
            "truncation": "reject", "question_group_size_max": 8,
            "score_composition": "independent-positive-negative-product",
            "packages": {x: version(x) for x in ("torch", "transformers", "safetensors", "huggingface-hub")},
            "snapshot_sha256": {str(p.relative_to(path)): file_sha(p) for p in sorted(path.rglob('*'))
                                if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts},
            "concurrency": "sequential model inference; CPU-heavy setup completed first",
        }

    def policies(self, state, tasks):
        import torch
        if len(self.encoder.text(self.encoding.render(state))) > self.encoder.max_state:
            raise ValueError("K2 state would be truncated")
        questions = {t.name: {"type": "choice", "instructions": t.instruction,
                             "criteria": t.labels, "label": next(iter(t.labels))} for t in tasks}
        for q in questions.values():
            options, _ = self.encoding.options_and_target(q)
            if any(len(self.encoder.text(o)) > 128 for o in options):
                raise ValueError("K2 option would be truncated")
        encoded = self.encoder.encode({"state": state, "questions": questions})
        if encoded is None or encoded["qkeys"] != list(questions) or len(encoded["decide"]) != len(tasks):
            raise ValueError("K2 questions exceed context")
        batch = self.encoding.collate([encoded], self.tokenizer.pad_token_id)
        tick = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(batch)
            policies = [s.float().softmax(-1).tolist() for s in logits]
        self.forward_seconds += time.perf_counter() - tick
        self.calls += 1
        self.tokens += len(encoded["ids"])
        return {t.name: dict(zip(t.labels, p)) for t, p in zip(tasks, policies)}

    def assess(self, world, candidates):
        tasks = build_tasks(candidates, PROMPT)
        scores = {}
        for offset in range(0, len(tasks), 8):
            scores.update(self.policies(WORLD_PREFIX + world, tasks[offset:offset + 8]))
        return decode_scores(tasks, scores)

    def resources(self):
        return {"forward_seconds": self.forward_seconds, "forward_calls": self.calls,
                "input_tokens": self.tokens}

    def close(self):
        pass


class LuxBackend(GGUFBackend):
    """Own a publisher-compatible Decision2 server; inherited resource accounting."""
    def __init__(self, path, binary, threads=4, port=8768):
        import subprocess
        import tempfile
        path, binary = Path(path).resolve(), Path(binary).resolve()
        if file_sha(path) != LUX_SHA:
            raise ValueError("Lux Q4_K_M checksum mismatch")
        self.url = f"http://127.0.0.1:{port}"
        self.http_seconds = self.http_calls = self.server_peak_rss_kib = 0
        self.log = tempfile.TemporaryFile()
        try:
            urllib.request.urlopen(self.url + "/health", timeout=1).close()
        except urllib.error.URLError:
            pass
        else:
            raise ValueError("Lux server port already in use")
        flags = ["--embeddings", "--pooling", "none", "-c", "8192", "-b", "8192",
                 "-ub", "8192", "-np", "1", "-t", str(threads), "-tb", str(threads),
                 "--no-context-shift", "--cache-ram", "0"]
        self.process = subprocess.Popen([str(binary), "-m", str(path), "--host", "127.0.0.1",
                                         "--port", str(port), *flags], stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    raise RuntimeError(self.log.read().decode(errors="replace")[-6000:])
                try:
                    props = self.request("props")
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Lux server startup timed out")
                    time.sleep(.2)
            if Path(props["model_path"]) != path:
                raise ValueError("Unexpected Lux model path")
            self._identity = {
                "backend": "lux-decision2-gguf", "model": "vllm-sr/Decision-2.0-Lux-9B-GGUF",
                "revision": LUX_REVISION, "quantization": "Q4_K_M", "sha256": LUX_SHA,
                "runtime_commit": RUNTIME_COMMIT, "build_info": props.get("build_info"),
                "server_binary_sha256": file_sha(binary), "server_flags": flags,
                "runtime_libraries_sha256": {p.name: file_sha(p) for p in sorted(binary.parent.glob("*.so*")) if p.is_file()},
                "prompt_version": PROMPT, "world_prefix": WORLD_PREFIX,
                "endpoint": "/v1/systemone", "readout": "native Decision2 head and GGUF temperature metadata",
                "option_order": ["no", "yes"],
                "attention": "publisher Decision2 single-question native requests; no inter-question attention",
                "truncation": "reject", "question_group_size_max": 8,
                "score_composition": "independent-positive-negative-product", "threads": threads,
                "concurrency": "sequential model inference; CPU-heavy setup completed first",
            }
        except BaseException:
            self.close()
            raise

    def assess(self, world, candidates):
        tasks = build_tasks(candidates, PROMPT)
        scores = {}
        for offset in range(0, len(tasks), 8):
            group = tasks[offset:offset + 8]
            payload = {"state": WORLD_PREFIX + world, "questions": {
                t.name: {"type": "choice", "instructions": t.instruction, "criteria": t.labels} for t in group}}
            response = self.request("v1/systemone", payload)
            if set(response["answers"]) != set(payload["questions"]) or response["usage"]["output_tokens"] != 0:
                raise ValueError("Lux response contract mismatch")
            for t in group:
                p = response["answers"][t.name]["probabilities"]
                if set(p) != set(t.labels) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()) or not math.isclose(sum(p.values()), 1., abs_tol=1e-6):
                    raise ValueError("Lux invalid probability distribution")
                scores[t.name] = p
        return decode_scores(tasks, scores)
