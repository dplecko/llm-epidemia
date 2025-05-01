from abc import ABC, abstractmethod
import torch
import evaluator_helpers
from openai import OpenAI
from google import genai
import string
import random


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
    def predict(self, prompt, levels, n_mc, max_batch_size):
        """
        Generate model predictions based on the prompt and levels.

        Args:
            model: Language model for generation.
            levels List[str]: Possible answers.
            n_mc (int): Total number of samples to generate. That is, permutations of the possible answers.
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
    
    def prepare_prompt(self, prompt, levels):
        """
        Prepare the prompt for the model.
        :param prompt: The initial prompt.
        :param levels: A list of possible answers.
        :return: The prepared prompt.
        """
        if levels is None:
            levels = self.genereate_levels()
        permuted_levels = shuffled_copy(levels)
        prompt = prompt + "\n" + "Begin your answer with the capital letter that corresponds to your chosen option, followed by a dot and a justification:\n"
        answers, answer_mapping = self.prepare_answers(permuted_levels)
        prompt += answers
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
    
    
    def predict(self, prompt, levels, n_mc, max_batch_size):
        average_probs = {level: [] for level in levels}
        for i in range(n_mc):
            processed_prompt, answer_mapping = self.prepare_prompt(prompt, levels)
            inputs = self.tokenizer(processed_prompt, return_tensors="pt").to(self.device)

            # Convert words in levels to token IDs
            level_ids = [[self.tokenizer.convert_tokens_to_ids(t) for t in lvl] for lvl in answer_mapping.keys()]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, 0, :]  # Get logits for the first token
                probs = torch.softmax(logits, dim=-1)

            level_probs = [sum(probs[0, tid].item() for tid in lvl_ids) for lvl_ids in level_ids]
            total_prob = sum(level_probs)
            prob_distr = [p / total_prob for p in level_probs]
            for i, answer in enumerate(answer_mapping.keys()):
                level = answer_mapping[answer]
                average_probs[level].append(prob_distr[i])
                
        return [sum(vals) / len(vals) for vals in average_probs.values()], None
    
    
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
    
    def predict(self, prompt, levels, n_mc, max_batch_size):
        samples = []
        for i in range(n_mc):
            processed_prompt, answer_mapping = self.prepare_prompt(prompt, levels)
            generated_text = self._sample(processed_prompt).strip()
            models_answer = generated_text[0]  # model has to start with A, B, C, D,...
            samples.append(answer_mapping.get(models_answer, None))  # None if not in mapping
            
        return samples, generated_text
    
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