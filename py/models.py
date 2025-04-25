from abc import ABC, abstractmethod
import torch
import evaluator_helpers
from openai import OpenAI


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
    def lvl_sample(self, prompt, levels, n_mc, max_batch_size):
        """
        Generate multiple completions and classify each to a level.

        Args:
            model: Language model for generation.
            tokenizer: Tokenizer with .generate and .batch_decode methods.
            inputs (dict): Tokenized input tensors.
            levels (List[List[str]]): List of levels for classification.
            n_mc (int): Total number of samples to generate.
            max_batch_size (int): Max samples per batch.

        Returns:
            List[Union[int, float, None]]: Level index or 0.5/None per sample.
        """
        pass
    
    def lvl_probs(self, prompt, levels):
        """
        Compute normalized probabilities for each factor level based on model's logits.

        Args:
            model: A language model that outputs logits.
            tokenizer: Tokenizer with convert_tokens_to_ids method.
            inputs (dict): Tokenized input tensors for the model.
            levels (List[List[str]]): List of levels, each a list of word strings.

        Returns:
            List[float] or None: Normalized probabilities over levels or None if levels is None.
        """
        pass
    
    @abstractmethod
    def cts_sample(self, prompt, n_mc, max_batch_size, max_tokens=10):
        """
        Generate numeric answers from text completions.

        Args:
            model: Language model for generation.
            tokenizer: Tokenizer used to decode outputs.
            inputs (dict): Tokenized input tensors.
            n_mc (int): Number of completions to sample.
            max_batch_size (int): Max samples per generation batch.
            max_tokens (int): Max tokens to generate per sample.

        Returns:
            List[float or None]: Extracted numeric values from completions.
        """
        pass
    
    @abstractmethod
    def story_sample(self, prompt, second_prompt, levels, n_mc, max_batch_size, max_tokens=50):
        """
        Generate a story, follow up with a question, and extract a level or numeric answer.

        Args:
            model: Language model for generation.
            tokenizer: Tokenizer for prompt construction and decoding.
            inputs (dict): Initial prompt inputs for story generation.
            second_prompt (str): Follow-up question to ask about the story.
            levels (List[List[str]] or None): Levels for classification (optional).
            n_mc (int): Number of samples to generate.
            max_batch_size (int): Max samples per generation batch.
            max_tokens (int): Max tokens for story generation.

        Returns:
            List[Union[int, float, None]]: Extracted level index or numeric value per sample.
        """
        pass
    
    @abstractmethod
    def get_type(self):
        """
        Get the type of model.
        :return: A string representing the model type.
        """
        pass


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
    
    
    def lvl_sample(self, prompt, levels, n_mc, max_batch_size):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        samples = []
        for i in range(0, n_mc, max_batch_size):
            batch_size = min(max_batch_size, n_mc - i)
            batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=30,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            # Extract generated tokens
            generated_tokens = self.tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Map text to levels
            samples.extend(evaluator_helpers.txt_to_lvl(token, levels) for token in generated_tokens)

        return samples, generated_tokens
    
    
    def lvl_probs(self, prompt, levels):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if levels is None:
            return None

        # Convert words in levels to token IDs
        level_ids = [[self.tokenizer.convert_tokens_to_ids(t) for t in lvl] for lvl in levels]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token
            probs = torch.softmax(logits, dim=-1)

        level_probs = [sum(probs[0, tid].item() for tid in lvl_ids) for lvl_ids in level_ids]
        total_prob = sum(level_probs)
        return [p / total_prob for p in level_probs], None
    
    
    def cts_sample(self, prompt, n_mc, max_batch_size, max_tokens=10):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        samples = []
        for i in range(0, n_mc, max_batch_size):
            batch_size = min(max_batch_size, n_mc - i)
            batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            # Extract generated text
            generated_texts = self.tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Convert text to numeric values
            samples.extend(evaluator_helpers.txt_to_num(text) for text in generated_texts)

        return samples, generated_texts
    
    
    def story_sample(self, prompt, second_prompt, levels, n_mc, max_batch_size, max_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        samples = []
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure padding token is set
        
        for i in range(0, n_mc, max_batch_size):
            batch_size = min(max_batch_size, n_mc - i)
            batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                generated_story = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            generated_texts = self.tokenizer.batch_decode(
                generated_story[:, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Construct the follow-up prompt
            followup_prompts = [
                f"Based on the following text: {text} answer the following question: {second_prompt}"
                for text in generated_texts
            ]

            # Tokenize follow-up prompts
            followup_inputs = self.tokenizer(followup_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                generated_responses = self.model.generate(
                    **followup_inputs,
                    max_new_tokens=3,  # Short answer expected
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Extract input length to remove from generated responses
            input_length = followup_inputs["input_ids"].shape[1]

            # Decode only the newly generated portion
            response_texts = self.tokenizer.batch_decode(generated_responses[:, input_length:], skip_special_tokens=True)

            for response in response_texts:
                if levels is not None:
                    samples.append(evaluator_helpers.txt_to_lvl(response, levels))
                else:
                    samples.append(evaluator_helpers.txt_to_num(response))

        return samples, response_texts
    
    def get_type(self):
        return "HuggingFace"
    
    

# An adapter for an API-based model (example implementation)
class OpenAIAPIModel(AbstractModel):
    def __init__(self, model_name, reasoning=None, tools=[]):
        """
        Initialize with an API client that communicates with the remote model.
        :param api_client: An object that knows how to interact with a remote model via API.
        """
        self.client = OpenAI()
        self.model_name = model_name
        self.reasoning = reasoning
        self.tools = tools

    def __call__(self, **inputs):
        # For example, send a POST request with the inputs.
        # This is pseudo-code; your actual API call implementation may vary.
        response = self.api_client.call_model(**inputs)
        return response

    def generate(self, **kwargs):
        # Forward the generate call to the API client.
        response = self.api_client.generate(**kwargs)
        return response
    
    def lvl_sample(self, prompt, levels, n_mc, max_batch_size):
        samples = []
        generated_text = self._sample(n_mc, prompt)
        samples.extend(evaluator_helpers.txt_to_lvl(token, levels) for token in generated_text)
        return samples, generated_text
    
    def lvl_probs(self, prompt, levels):
        raise NotImplementedError("APIModel does not support level probabilities.")
    
    def _sample(self, n_mc, prompt):
        outputs = []
        for i in range(n_mc):
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                reasoning=self.reasoning,
                tools=self.tools,
            )
            outputs.append(response.output_text)
        return outputs
    
    def cts_sample(self, prompt, n_mc, max_batch_size, max_tokens=10):
        samples = []
        generated_text = self._sample(n_mc, prompt)
        samples.extend(evaluator_helpers.txt_to_num(text) for text in generated_text)
        return samples, generated_text
    
    def story_sample(self, prompt, second_prompt, levels, n_mc, max_batch_size, max_tokens=50):
        samples = []
        generated_texts = self._sample(n_mc, prompt)
        
        followup_prompts = [
            f"Based on the following text: {text} answer the following question: {second_prompt}"
            for text in generated_texts
        ]
        
        response_texts = self._sample(n_mc, followup_prompts)
        
        for response in response_texts:
            if levels is not None:
                samples.append(evaluator_helpers.txt_to_lvl(response, levels))
            else:
                samples.append(evaluator_helpers.txt_to_num(response))
        
        return samples, response_texts
    
    def get_type(self):
        return "API"