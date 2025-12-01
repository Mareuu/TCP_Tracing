"""
TCP LLM Service - Language Model Serving for TCP Pipeline
Provides LLM inference using SGLang backend.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tenacity import retry, wait_exponential, wait_random
import subprocess
import time
import requests
import copy


def init_sglang_serv(model_path, model_len=30000, seed=0, n_gpu=1, temperature=1., max_timeout=60*60*3, min_p=0.05, max_memory=0.92, fp8=False, repetition_penalty=0, top_p=1., top_k=-1, max_running_requests=0, schedule_conservativeness=-1, enable_thinking=False, max_tokens=-1):
    from openai import OpenAI
    import numpy as np
    import requests

    from sglang.utils import (
        execute_shell_command,
        wait_for_server,
    )

    port = np.random.randint(30100, 30200)

    def launch_serv(model_path, model_len, seed, n_gpu, max_memory, fp8, port, max_running_requests=max_running_requests, schedule_conservativeness=schedule_conservativeness, disable_overlap=False):
        command = f"python -m sglang.launch_server --model-path {model_path} --port {port} --host 0.0.0.0 --tp {n_gpu} --context-length {model_len} --random-seed {seed} --mem-fraction-static {max_memory} "
        print("schedule_conservativeness", schedule_conservativeness)
        print("max_running_requests", max_running_requests)
        if "qwen3" in model_path.lower():
            command += "--reasoning-parser qwen3 "
        if fp8:
            command += "--quantization fp8 "
        if max_running_requests != 0:
            command += f"--max-running-requests {max_running_requests} "
        if schedule_conservativeness != -1:
            command += f"--schedule-conservativeness {schedule_conservativeness} "
        if disable_overlap:
            command += "--disable-overlap-schedule "
        server_process = execute_shell_command(command)
        print(command)
        return server_process

    def check_server_run(model_path, port, server_process):
        """Check if the server is running and serving the correct model.
        Needed when launching multiple inference processes on a cluster.
        """
        try:
            wait_for_server(f"http://localhost:{port}")
            time.sleep(15)
            response = requests.get(
                f"http://localhost:{port}/get_model_info",
                headers={"Authorization": "Bearer None"},
            )
            good_model = response.json()["model_path"] == model_path
            if not good_model:
                raise Exception("wrong model")
        except:
            return False
        is_running = server_process.poll() is None
        if not is_running:
            return False
        return True

    disable_overlap = False
    if repetition_penalty > 0:
        disable_overlap = True
    server_process = launch_serv(model_path, model_len, seed, n_gpu, max_memory, fp8, port, max_running_requests=max_running_requests, schedule_conservativeness=schedule_conservativeness, disable_overlap=disable_overlap)

    is_running = check_server_run(model_path, port, server_process)
    if not is_running:
        for _ in range(10):
            port += 1
            server_process = launch_serv(model_path, model_len, seed, n_gpu, max_memory, fp8, port, max_running_requests=max_running_requests, schedule_conservativeness=schedule_conservativeness, disable_overlap=disable_overlap)
            is_good = check_server_run(model_path, port, server_process)
            if is_good:
                break

    is_good = check_server_run(model_path, port, server_process)
    if not is_good:
        raise Exception("wrong model")

    client = OpenAI(
        timeout=max_timeout,
        api_key="None",
        base_url=f"http://localhost:{port}/v1"
    )
    cfg_generation = {
        "temperature": temperature,
        "model": model_path,
        "extra_body": {"min_p": min_p}
    }
    if top_p != 1.0:
        cfg_generation["top_p"] = top_p
    if top_k != -1:
        cfg_generation["extra_body"]["top_k"] = top_k
    if repetition_penalty != 0:
        cfg_generation["frequency_penalty"] = repetition_penalty
    if "qwen3" in model_path.lower():
        cfg_generation["extra_body"]["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if max_tokens != -1:
        cfg_generation["max_tokens"] = max_tokens
    return server_process, client, cfg_generation


def terminate_sglang_serv(server_process):
    from sglang.utils import terminate_process
    terminate_process(server_process)


class LLM_serv:
    """
    Language Model Service for TCP pipeline.

    Provides LLM inference using SGLang backend with OpenAI-compatible API.

    Args:
        model_path: HuggingFace model identifier or local path
        model_len: Maximum context length
        seed: Random seed for reproducibility
        n_gpu: Number of GPUs (tensor parallel size)
        temperature: Sampling temperature
        max_timeout: Maximum timeout for generation in seconds
        min_p: Minimum probability threshold for sampling
        max_workers: Maximum concurrent workers for batch processing
        gpu_mem: GPU memory utilization ratio (0.0-1.0)
        fp8: Whether to use FP8 quantization
        repetition_penalty: Repetition penalty (0 to disable)
        top_p: Top-p (nucleus) sampling parameter
        top_k: Top-k sampling parameter (-1 to disable)
        max_running_requests: Max concurrent requests (0 for default)
        schedule_conservativeness: Scheduling parameter (-1 for default)
        enable_thinking: Enable thinking mode for supported models (e.g., Qwen3)
        max_tokens: Maximum tokens to generate (-1 for default)
    """

    def __init__(self, model_path, model_len=30000, seed=0, n_gpu=1, temperature=1., max_timeout=60*60*3, min_p=0.05, max_workers=64, gpu_mem=0.9, fp8=False, repetition_penalty=0, top_p=1., top_k=-1, max_running_requests=0, schedule_conservativeness=-1, enable_thinking=False, max_tokens=-1):
        self.model_path = model_path
        self.model_len = model_len
        self.seed = seed
        self.n_gpu = n_gpu
        self.temperature = temperature
        self.max_timeout = max_timeout
        self.min_p = min_p
        self.max_workers = max_workers
        print("max_running_requests", max_running_requests)
        print("schedule_conservativeness", schedule_conservativeness)
        self.server_process, self.client, self.cfg_generation = init_sglang_serv(
            model_path, model_len, seed, n_gpu, temperature, max_timeout, min_p,
            max_memory=gpu_mem, fp8=fp8, repetition_penalty=repetition_penalty,
            top_p=top_p, top_k=top_k, max_running_requests=max_running_requests,
            schedule_conservativeness=schedule_conservativeness,
            enable_thinking=enable_thinking, max_tokens=max_tokens
        )
        out = self.generate(["Hello"])
        print(out)

    def generate(self, prompts, n=1):
        """
        Generate completions for the given prompts.

        Args:
            prompts: List of prompt strings
            n: Number of completions per prompt

        Returns:
            List of lists of generated texts
        """
        out = get_multiple_completions(self.client, prompts, self.cfg_generation, max_workers=self.max_workers, n=n)
        return out

    def terminate(self):
        """Terminate the SGLang server process."""
        terminate_sglang_serv(self.server_process)


@retry(wait=wait_exponential(multiplier=1, min=30, max=600) + wait_random(min=0, max=1))
def get_completion(client, prompt: str, cfg_generation, system_prompt=None, temperature=None, n=1) -> list[str]:
    """Get completion(s) from OpenAI-compatible API"""
    kwargs = cfg_generation.copy()
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["n"] = n
    if system_prompt is None:
        system_prompt = "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by reasoning and generating Python code."

    if "deepseek" in cfg_generation.get("model") or "google" in cfg_generation.get("model"):
        out = []
        n = copy.deepcopy(kwargs["n"])
        kwargs["n"] = 1
        for _ in range(n):
            try:
                completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    **kwargs
                )
                out.append(completion.choices[0].message.content)
            except Exception as e:
                print("completion problem: ", e)
                return [None] * n
        return out
    else:
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
        except Exception as e:
            print("completion problem: ", e)
            too_long = "longer than the model's context length" in e.body["message"]
            if too_long:
                return [e.body["message"]] * n
            return [None] * n

        out = [choice.message.content for choice in completion.choices]
        return out


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_multiple_completions(client, batch_prompt: list[str], cfg_generation: dict = {}, batch_tools: list[list[dict]] = None, max_workers=20, temperature=None, n=1, timeout_seconds=None) -> list[list[str]]:
    """Get multiple completions from OpenAI-compatible API"""
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]

    completions = []
    count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_batch in chunks(batch_prompt, max_workers):
            for message in sub_batch:
                count += 1
                kwargs = {
                    "client": client,
                    "prompt": message,
                    "cfg_generation": cfg_generation,
                    "temperature": temperature,
                    "n": n
                }
                future = executor.submit(get_completion, **kwargs)
                completions.append(future)
            time.sleep(5)
            print(f"send {count} / {len(batch_prompt)} messages")

    # Retrieve the results from the futures
    if timeout_seconds is None:
        out_n = [future.result() for future in completions]
    else:
        out_n = []
        for future in completions:
            try:
                result = future.result(timeout=timeout_seconds)
                out_n.append(result)
            except TimeoutError:
                print(f"A call to get_completion timed out after {timeout_seconds} seconds")
                out_n.append(None)
            except Exception as e:
                print(f"An error occurred: {e}")
                out_n.append(None)
    return out_n
