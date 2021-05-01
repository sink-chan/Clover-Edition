import os
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import re
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from getconfig import settings, logger
from utils import cut_trailing_sentence, output, clear_lines, format_result, use_ptoolkit

if not settings.getboolean('force-cpu') and not torch.cuda.is_available():
    logger.warning('CUDA is not available, you are limited to CPU only.')

DTYPE = torch.float32 if ((not torch.cuda.is_available()) or settings.getboolean('force-cpu')) else torch.float16
logger.info('Cuda Available: {}    Force CPU: {}    Precision: {}'.format(torch.cuda.is_available(),
                                                                          settings.getboolean('force-cpu'),
                                                                          '32-bit' if DTYPE == torch.float32 else '16-bit'))

# warnings.filterwarnings("ignore")
MODEL_CLASSES = {
    "gpt_neo": (GPTNeoForCausalLM,GPT2Tokenizer),
}

def memory_merge(prompt, context, tokenizer, device, maxHistory=2048):
    assert (prompt + context)

    ids = tokenizer(prompt + "\n" + context, return_tensors="pt", add_prefix_space=True, add_special_tokens=False,).input_ids.to(device)
    print(ids)

    # Truncate to max length if needed.
    ids = ids[-maxHistory:]

    #if ids.shape[1] > maxHistory:
    #    logger.error("CONTEXT IS TOO LONG ERROR")
    #    ids = ids[-maxHistory:]
    return ids


# length should be max length, other settings should be removed, device should not be set
# we could possibly optimize this by having larger batch sizes but it would likely double or more the memory requirements
def sample_sequence(
        model,
        length,
        context,
        temperature=1,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        device="cpu",
        stop_tokens=None,
        tokenizer=None
):
    """Actually generate the tokens"""
    logger.debug(
        'temp: {}    top_k: {}    top_p: {}    rep-pen: {}'.format(temperature, top_k, top_p, repetition_penalty))

    max_length = context.shape[1] + length # check to see if greater than 2048?

    if settings.getboolean('force-cpu'):
        context = context.long().cpu()
    else:
        context = context.long().cuda()

    out = model.generate(
        context,
        do_sample=True,
        min_length=max_length,
        max_length=max_length,
        temperature=temperature,
        top_k = top_k,
        top_p = top_p,
        repetition_penalty = repetition_penalty,
        repetition_penalty_range = 300,
        repetition_penalty_slope = 3.33,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    ).long()

    generated = tokenizer.decode(out[0])

    return generated


class GPTNeoGenerator:
    def __init__(
            self, generate_num=60, temperature=0.4, top_k=40, top_p=0.9, dtype=DTYPE,
            model_path: Union[str, Path]=Path('models', 'gpt-neo-2.7B-horni'), repetition_penalty=1,
    ):
        self.generate_num = generate_num
        self.temp = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.samples = 1
        self.dtype = dtype
        self.repetition_penalty = repetition_penalty
        self.batch_size = 1
        self.max_history_tokens = 1024 - generate_num
        self.stop_token = "<|endoftext|>"

        if isinstance(model_path, str):
            self.checkpoint_path = model_path
            logger.warning(
                f"Using DEBUG MODE! This will load one of the generic (non-finetuned) GPT2 models. "
                f"Selected: {model_path}")
        elif isinstance(model_path, Path):
            self.checkpoint_path = model_path
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    "Could not find {} Make sure to download a pytorch model and put it in the models directory!".format(
                        str(self.checkpoint_path)))
        else:
            raise ValueError(f"model_path must be either str or Path, got {type(model_path)}")

        self.device = torch.device("cuda" if self.dtype == torch.float16 else "cpu")
        logger.info(
            "Using device={}, checkpoint={}, dtype={}".format(self.device, str(self.checkpoint_path), self.dtype))

        # Load tokenizer and model
        model_class, tokenizer_class = MODEL_CLASSES["gpt_neo"]
        self.checkpoint = torch.load(Path(model_path, 'pytorch_model.bin'), map_location='cpu')
        self.tokenizer = tokenizer_class.from_pretrained(Path(model_path))
        self.model = model_class.from_pretrained(model_path, state_dict=self.checkpoint)
        self.model.to(self.dtype).to(self.device)
        self.model.eval()

    def sample_sequence(
            self, context_tokens=None, top_k=None, top_p=None, repetition_penalty=None, generate_num=None,
            temperature=None, stop_tokens=None
    ):
        assert (top_k is not None)
        assert (temperature is not None)
        assert (top_p)
        assert (repetition_penalty)
        generate_num = generate_num if (generate_num is not None) else self.generate_num
        temperature = temperature if (temperature is not None) else self.temp
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        out = sample_sequence(
            model=self.model,
            context=context_tokens,
            length=generate_num,
            # context=self.context,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=self.device,
            stop_tokens=stop_tokens,
            tokenizer=self.tokenizer
            # batch_size=self.batch_size,
        )
        return out

    def result_replace(self, result, allow_action=False):
        # logger.debug("BEFORE RESULT_REPLACE: `%s`", repr(result))

        result = cut_trailing_sentence(result, allow_action=allow_action)

        if len(result) == 0:
            return ""
        first_letter_capitalized = result[0].isupper()
        result = result.replace('."', '".')
        result = result.replace("#", "")
        result = result.replace("*", "")
        # TODO look at this I think blank lines should be fine or blacklisted at generation time
        result = result.replace("\n\n", "\n")
        # result = first_to_second_person(result)

        if not first_letter_capitalized:
            result = result[0].lower() + result[1:]

        # this is annoying since we can already see the AIs output
        # logger.debug( "AFTER RESULT_REPLACE: `%r`. allow_action=%r", repr(result), allow_action)

        return result

    def generate_raw(
            self, context, prompt='', generate_num=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, stop_tokens=None
    ):
        assert (top_k is not None)
        assert (temperature is not None)
        assert (top_p)
        assert (repetition_penalty)

        context_tokens = memory_merge(prompt, context, self.tokenizer, self.device, self.max_history_tokens)

        #logger.debug(
        #    "Text passing into model `%r`",
        #    self.tokenizer.decode(
        #        context_tokens,
                #clean_up_tokenization_spaces=True,
                # skip_special_tokens=True,
        #    ),
        #)
        generated = 0
        text = ""
        for _ in range(self.samples // self.batch_size):
            out = self.sample_sequence(
                context_tokens,
                generate_num=generate_num,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_tokens=stop_tokens,
            )
            #print(out)
            text += out
            generated += 1
            # disabled clean up of spaces, see what effect this has TODO
            if self.stop_token:
                index = text.find(self.stop_token)
                if index == -1:
                    index = None
                text = text[:index]
            if stop_tokens is not None:
                for stop_token in stop_tokens:
                    index = text.find(self.stop_token)
                    if index == -1:
                        index = None
                    text = text[:index]
        return text

    def generate(self, context, prompt='', temperature=None, top_p=None, top_k=None, repetition_penalty=None, depth=0):
        assert (top_k is not None)
        assert (temperature is not None)
        assert (top_p)
        assert (repetition_penalty)
        # logger.debug("BEFORE PROMPT_REPLACE: `%r`", prompt)

        # prompt = [self.prompt_replace(p) for p in prompt]

        # logger.debug("AFTER PROMPT_REPLACE is: `%r`", repr(prompt))
        assert (prompt + context)

        text = self.generate_raw(
            context, prompt, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            stop_tokens=torch.tensor([[self.tokenizer.eos_token_id]])
        )

        logger.debug("Generated result is: `%r`", repr(text))

        result = self.result_replace(text)

        if (depth > 6) and len(result) == 0:
            # Sometimes it keeps generating a story startng with an action (">"), if it's tried a few times and it keeps
            # happening, lets let it keep action text which starts in ">"
            # We could just blacklist that token and force it to generate something else. TODO
            result = self.result_replace(text, allow_action=True)
            logger.info(
                "Model generated empty text after formatting `%r`. Trying to format less with allow_action=True. `%r`",
                text,
                result,
            )

            # same here as above
        if len(result) == 0:
            if depth < 20:
                logger.info("Model generated empty text trying again %r", depth)
                return self.generate(
                    prompt, context, temperature=temperature, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty, depth=depth + 1
                )
            else:
                logger.warn(
                    "Model generated empty text %r times. Try another action", depth
                )
        return result
