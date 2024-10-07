# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# [2024] CHANGES MADE BY Yichen Gong, Delong Ran. Licensed under the Apache License 2.0, see LICENSE file.

import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lm_eval import base, evaluator, tasks
from lm_eval.base import BaseLM

from litgpt import PromptStyle, Tokenizer

from litgpt.adapter import GPT, Config
from litgpt.generate.base import generate
from litgpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint, lazy_load,load_config, get_model_path, get_dataset_info, resolve_output_file
import time

def setup_seed(seed):
     import numpy as np
     import random
     import torch
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)
class EvalHarnessBase(BaseLM):
    # Credits:
    # https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py
    def __init__(self, fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, batch_size: int):
        super().__init__()
        self.fabric = fabric
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size
        with fabric.init_tensor():
            model.set_kv_cache(batch_size=batch_size)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        kwargs = {el.split("=")[0]: el.split("=")[1] for el in arg_string.split(",")}
        return cls(**kwargs, **additional_config)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self.model.max_seq_length

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu * self.fabric.world_size

    @property
    def device(self):
        return self.fabric.device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, bos=False, eos=False).tolist()

    def tok_decode(self, tokens: List[int]) -> str:
        t = torch.tensor(tokens)
        return self.tokenizer.decode(t)

    @torch.inference_mode()
    def _model_call(self, inps):
        return self.model(inps)

    @torch.inference_mode()
    def _model_generate(self, context, max_length, eos_token_id) -> torch.Tensor:
        # this only supports batch size 1
        assert context.shape[0] == 1
        out = generate(self.model, context[0], max_length, eos_id=eos_token_id)
        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        return out.unsqueeze(0)

    @torch.inference_mode()
    def run_eval(
        self, eval_tasks: List[str], num_fewshot: int, limit: Optional[int], bootstrap_iters: int, no_cache: bool
    ) -> Dict:
        # Returns a list containing all values of the task registry that
        # match at least one of the patterns
        import fnmatch

        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            return list(task_names)

        eval_tasks = pattern_match(eval_tasks, tasks.ALL_TASKS)
        print(f"Found tasks: {eval_tasks}")

        # **HACK INCOMING**:
        # first get task dict on local main rank
        # the tasks are downloaded *as they are initialized*, and the downloads don't like multithreading.
        # so we download them once on the local main rank, wait, and then initialize them on all other ranks, which *should* load from the cache.
        if self.fabric.local_rank == 0:
            tasks.get_task_dict(eval_tasks)
        # torch barrier
        self.fabric.barrier()
        tasks.get_task_dict(eval_tasks)

        lm = self
        if not no_cache:
            lm = base.CachingLM(lm, "lm_cache/litgpt.db")

        results = evaluator.evaluate(
            lm=lm,
            task_dict=tasks.get_task_dict(eval_tasks),
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )
        results["config"] = dict(
            model=self.model.config.name,
            batch_size=self.batch_size,
            device=str(self.device),
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            no_cache=no_cache,
        )
        return results


@torch.inference_mode()
def run_eval_harness(
    adapter_dir: Path,
    model_name: str = "Llama-2-7b-chat-hf",
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    eval_tasks: List[str] = ["hellaswag","boolq","arc_easy"],
    num_fewshot: int = 0,
    limit: Optional[int|float] = 0.1,
    bootstrap_iters: int = 100000,
    no_cache: bool = True,
    output_file: Optional[str] = None
):
    result_path = resolve_output_file(output_file)
    
    base_model_config = load_config('configs/base_model_path.yaml')
    original_model_name_or_path = get_model_path(model_name, base_model_config)
    checkpoint_dir = Path(original_model_name_or_path)
    
    adapter_path = adapter_dir / "lit_model.pth"
    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=[0], precision=precision, plugins=plugins)
    fabric.launch()
    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)


    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = 512
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    adapter_checkpoint = lazy_load(adapter_path)

    checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model = fabric.setup(model)


    eval_harness = EvalHarnessBase(fabric, model, tokenizer, 1)
    print(eval_tasks, num_fewshot, limit, bootstrap_iters, no_cache)
    results = eval_harness.run_eval(eval_tasks, num_fewshot, limit, bootstrap_iters, no_cache)
    print(results)
    if result_path is not None:
        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(results,file, indent=2)




if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    CLI(run_eval_harness)