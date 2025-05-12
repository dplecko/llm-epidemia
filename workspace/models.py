from abc import ABC, abstractmethod
import torch
import evaluator_helpers
from openai import OpenAI
# from google import generativeai as genai
import string
import random
import math
import itertools
from collections import defaultdict


def shuffled_copy(items):
    """
    Return a random permutation of *items* without modifying the original list.

    >>> original = ["alpha", "beta", "gamma", "delta"]
    >>> shuffled = shuffled_copy(original)
    >>> shuffled                        # random order
    ['gamma', 'alpha', 'delta', 'beta']
    >>> original                        # unchanged
    ['alpha', 'beta', 'gamma', 'delta']
    """
    # random.sample creates a brand-new list of the requested length
    return random.sample(items, k=len(items))


# Abstract base model defining the common interface
class AbstractModel(ABC):
    @abstractmethod
    def __call__(self, **inputs):
        """
        Call the model with the given tokenized inputs.
        :param inputs: A dict of tokenized input tensors.
        :return: Model outputs.
        """
        pass

    @abstractmethod
    def generate(self, **kwargs):
        """
        Generate outputs from the model using given parameters.
        :param kwargs: A dict that must contain tokenized inputs along with any generation-related parameters.
        :return: Generated sequences or results.
        """
        pass
    
    @abstractmethod
    def predict(self, prompt, levels, num_permutations, max_batch_size):
        """
        Generate model predictions based on the prompt and levels.

        Args:
            model: Language model for generation.
            levels List[str]: Possible answers.
            num_permutations (int): Total number of samples to generate. That is, permutations of the possible answers.
            max_batch_size (int): Max samples per batch.

        Returns:
            List[Union[int, float, None]]: Level index or 0.5/None per sample.
        """
        pass

    
    @abstractmethod
    def get_type(self):
        """
        Get the type of model.
        :return: A string representing the model type.
        """
        pass
    
    def prepare_answers(self, levels):
        """
        Prepare the answers for the model.
        :param levels: A list of possible answers.
        :return: A list of tokenized answers.
        """
        if len(levels) > 26:
            raise ValueError("Supports up to 26 items (A-Z)")

        letters = string.ascii_uppercase                # 'A', 'B', ...
        mapping = {letters[i]: item                     # {'A': 'first', ...}
                for i, item in enumerate(levels)}

        answer_key = "\n".join(f"{k}. {v}"              # 'A. first\nB. second'
                            for k, v in mapping.items())

        return answer_key, mapping
    
    def genereate_levels(self):
        """
        Generate a list of possible answers.
        :return: A list of possible answers.
        """
        # TODO
        pass
    
    def prepare_prompt(self, prompt, levels, given_permutation=None):
        """
        Prepare the prompt for the model.
        :param prompt: The initial prompt.
        :param levels: A list of possible answers.
        :param given_permutation: A a provided permutation.
        :return: The prepared prompt.
        """
        if levels is None:
            levels = self.genereate_levels()
        if given_permutation is None:
            permuted_levels = shuffled_copy(levels)
        else:
            permuted_levels = given_permutation
        prompt = "Input: " + prompt + "\nBegin your answer with the capital letter corresponding to your chosen option below, followed by a dot and a justification."
        answers, answer_mapping = self.prepare_answers(permuted_levels)
        prompt += answers
        prompt += "\nOutput: "
        return prompt, answer_mapping


# An adapter for a Hugging Face model
class HuggingFaceModel(AbstractModel):
    def __init__(self, hf_model, tokenizer):
        """
        Initialize with a Hugging Face model.
        :param hf_model: A huggingface transformers model which supports __call__ and generate.
        """
        self.model = hf_model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def __call__(self, **inputs):
        # Directly call the underlying Hugging Face model with tokenized inputs.
        return self.model(**inputs)


    def generate(self, **kwargs):
        # Use the underlying Hugging Face generate method.
        # kwargs must include the tokenized batch inputs along with generation parameters.
        return self.model.generate(**kwargs)
    
    
    def predict(self, prompt, levels, num_permutations, max_batch_size):
        """
        Compute average probabilities for every `level` in `levels`
        using either a fixed number of random permutations
        or the full factorial set if it fits into memory.
        """
        # ------------------------------------------------------------------ #
        # Small helper ‑‑ keeps all model / tokenizer calls in one place.
        # ------------------------------------------------------------------ #
        def _evaluate_once(proc_prompt, answer_mapping, avg_probs):
            """
            Run the model on a single prompt and accumulate
            the resulting probability mass into `avg_probs`.
            """
            inputs = self.tokenizer(proc_prompt, return_tensors="pt").to(self.device)

            # Map each answer string (or tuple of tokens) to token IDs
            level_ids = [
                [self.tokenizer.convert_tokens_to_ids(tok) for tok in ans]
                for ans in answer_mapping.keys()
            ]

            with torch.no_grad():
                outputs = self.model(**inputs).logits
                next_token_logits = outputs[:, -1, :]  # Last token in the input sequence
                probs = torch.softmax(next_token_logits, dim=-1)

            # Normalise probability mass over the provided answers
            level_probs = [
                sum(probs[0, tid].item() for tid in ids) for ids in level_ids
            ]

            total_prob = sum(level_probs)
            prob_dist  = [p / total_prob for p in level_probs]

            # Accumulate
            for p, answer in zip(prob_dist, answer_mapping.keys()):
                level = answer_mapping[answer]
                avg_probs[level].append(p)

        # ------------------------------------------------------------------ #
        # Main body
        # ------------------------------------------------------------------ #
        average_probs = {lvl: [] for lvl in levels}
        n_fact        = math.factorial(len(levels))

        # Decide which permutations to iterate over
        if n_fact > num_permutations:           # sample `num_permutations` times
            permutation_iter = (None for _ in range(num_permutations))
        else:                                   # exhaustively iterate all permutations
            permutation_iter = itertools.permutations(levels)

        for perm in permutation_iter:
            if perm is None:
                processed_prompt, answer_map = self.prepare_prompt(prompt, levels)
            else:
                processed_prompt, answer_map = self.prepare_prompt(prompt, levels, perm)

            _evaluate_once(processed_prompt, answer_map, average_probs)

        # Average accumulated probabilities and return
        return levels, [sum(vals) / len(vals) for vals in average_probs.values()], None
    
    
    def predict_batch(self, prompts, levels, num_permutations, max_batch_size):
        average_probs = [defaultdict(list) for _ in prompts]
        n_fact        = math.factorial(len(levels))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # because of batch padding
        self.tokenizer.padding_side = "left" 
        if n_fact > num_permutations:           # sample `num_permutations` times
            permutation_iter = (None for _ in range(num_permutations))
        else:                                   # exhaustively iterate all permutations
            permutation_iter = itertools.permutations(levels)
            
        for perm in permutation_iter:
            processed_prompts, answer_maps = [], []
            for pr in prompts:
                if perm is None:
                    p_prompt, a_map = self.prepare_prompt(pr, levels)  # type: ignore[attr‑defined]
                else:
                    p_prompt, a_map = self.prepare_prompt(pr, levels, perm)  # type: ignore[attr‑defined]
                processed_prompts.append(p_prompt)
                answer_maps.append(a_map)
            
            for start in range(0, len(processed_prompts), max_batch_size):
                batch_prompts = processed_prompts[start : start + max_batch_size]
                batch_answer_maps = answer_maps[start : start + max_batch_size]
                
                max_length_in_batch = max(len(p) for p in batch_prompts)

                # 2.1 Tokenise
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_in_batch).to(self.device)
                # 2.2 Forward → (B, seq, vocab)
                with torch.no_grad():
                    logits = self.model(**inputs).logits  # type: ignore[attr‑defined]
                next_token_logits = logits[:, -1, :]  # (B, vocab)
                probs = torch.softmax(next_token_logits, dim=-1)

                # 2.3 Map answers → token ids (once per batch)
                batch_level_ids = [
                    [
                        [self.tokenizer.convert_tokens_to_ids(tok) for tok in ans]  # type: ignore[attr‑defined]
                        for ans in amap.keys()
                    ]
                    for amap in batch_answer_maps
                ]

                # 2.4 Accumulate probabilities
                for idx_b, (prob_row, lvl_ids, amap) in enumerate(
                    zip(probs, batch_level_ids, batch_answer_maps)
                ):
                    level_probs = [sum(prob_row[tid].item() for tid in ids) for ids in lvl_ids]
                    total = sum(level_probs)
                    prob_dist = [p / total for p in level_probs]

                    global_idx = start + idx_b
                    for p, answer in zip(prob_dist, amap.keys()):
                        level_idx = amap[answer]  # **expected to be an int index**
                        average_probs[global_idx][level_idx].append(p)

        # Final averaging – keep output order identical to ``levels``
        # averaged_distributions = []
        # for avg_dict in average_probs:
        #     for i in range(len(levels)):
        #         if not avg_dict[i]:
        #             raise RuntimeError(
        #                 f"No probability mass collected for level index {i}. "
        #                 "This should not happen – please check `prepare_prompt`."
        #             )
        #     averaged_distributions.append([
        #         sum(avg_dict[i]) / len(avg_dict[i]) for i in range(len(levels))
        #     ])
        
        averaged_distributions = []
        for item_dict in average_probs:
            current_dict_averages = []
            for key in item_dict.keys():
                value_list = item_dict[key]  # Get the list for the current key

                # Calculate average, handling the case where the list might be empty
                avg = sum(value_list) / len(value_list) if value_list else 0
                current_dict_averages.append(avg)

            averaged_distributions.append(current_dict_averages)
        return levels, averaged_distributions, None
        
    
    def get_type(self):
        return "HuggingFace"
    
    
    
# An adapter for an API-based model (example implementation)
class APIModel(AbstractModel):
    def __init__(self,):
        """
        Initialize with an API client that communicates with the remote model.
        :param api_client: An object that knows how to interact with a remote model via API.
        """
        super().__init__()


    def __call__(self, **inputs):
        # For example, send a POST request with the inputs.
        # This is pseudo-code; your actual API call implementation may vary.
        response = self.api_client.call_model(**inputs)
        return response

    def generate(self, **kwargs):
        # Forward the generate call to the API client.
        response = self.api_client.generate(**kwargs)
        return response
    
    @abstractmethod
    def _sample(self, prompt):
        pass
    
    def predict(self, prompt, levels, num_permutations, max_batch_size):
        samples = []
        for i in range(num_permutations):
            processed_prompt, answer_mapping = self.prepare_prompt(prompt, levels)
            generated_text = self._sample(processed_prompt).strip()
            models_answer = generated_text[0]  # model has to start with A, B, C, D,...
            samples.append(answer_mapping.get(models_answer, None))  # None if not in mapping
            
        return samples, [1] * len(samples), generated_text
    
    def get_type(self):
        return "API"
    
    

# An adapter for an API-based model (example implementation)
class OpenAIAPIModel(APIModel):
    def __init__(self, model_name, reasoning=None, tools=[]):
        """
        Initialize with an API client that communicates with the remote model.
        :param api_client: An object that knows how to interact with a remote model via API.
        """
        super().__init__()
        self.client = OpenAI()
        self.model_name = model_name
        self.reasoning = reasoning
        self.tools = tools
    
    def _sample(self, prompt):
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            reasoning=self.reasoning,
            tools=self.tools,
        )
        return response.output_text
    
    
# An adapter for an API-based model (example implementation)
class GeminiAPIModel(APIModel):
    def __init__(self, model_name, ):
        super().__init__()
        self.client = genai.Client()
        self.model_name = model_name
    
    def _sample(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_name, 
            contents=prompt,
            )
        return response.text