from abc import ABC, abstractmethod



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
    def lvl_sample(self, model, tokenizer, inputs, levels, n_mc, max_batch_size):
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
    
    def lvl_probs(self, tokenizer, inputs, levels):
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
    def cts_sample(model, tokenizer, inputs, n_mc, max_batch_size, max_tokens=10):
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
    def story_sample(model, tokenizer, inputs, second_prompt, levels, n_mc, max_batch_size, max_tokens=50):
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


# An adapter for a Hugging Face model
class HuggingFaceModel(AbstractModel):
    def __init__(self, hf_model, tokenizer):
        """
        Initialize with a Hugging Face model.
        :param hf_model: A huggingface transformers model which supports __call__ and generate.
        """
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def __call__(self, **inputs):
        # Directly call the underlying Hugging Face model with tokenized inputs.
        return self.hf_model(**inputs)

    def generate(self, **kwargs):
        # Use the underlying Hugging Face generate method.
        # kwargs must include the tokenized batch inputs along with generation parameters.
        return self.hf_model.generate(**kwargs)
    

# An adapter for an API-based model (example implementation)
class APIModel(AbstractModel):
    def __init__(self, api_client):
        """
        Initialize with an API client that communicates with the remote model.
        :param api_client: An object that knows how to interact with a remote model via API.
        """
        self.api_client = api_client

    def __call__(self, **inputs):
        # For example, send a POST request with the inputs.
        # This is pseudo-code; your actual API call implementation may vary.
        response = self.api_client.call_model(**inputs)
        return response

    def generate(self, **kwargs):
        # Forward the generate call to the API client.
        response = self.api_client.generate(**kwargs)
        return response