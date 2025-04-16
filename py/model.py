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


# An adapter for a Hugging Face model
class HuggingFaceModel(AbstractModel):
    def __init__(self, hf_model):
        """
        Initialize with a Hugging Face model.
        :param hf_model: A huggingface transformers model which supports __call__ and generate.
        """
        self.hf_model = hf_model

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