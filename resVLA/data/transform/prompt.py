from . import BaseTransform


class InjectDefaultPrompt(BaseTransform):
    """
    Inject a default prompt into the input data.

    :param data: A dictionary containing the data to be transformed. The prompt will be injected into the "prompt" key.
    :return: The input data with the default prompt injected.
    """

    def __init__(self, default_prompt: str = "Do task."):
        super().__init__()
        self.default_prompt = default_prompt

    def __call__(self, data):
        if "prompt" not in data:
            data["prompt"] = self.default_prompt
        return data


class CoTPrompt(BaseTransform):
    """
    Inject a chain-of-thought (CoT) prompt into the input data.

    :param data: A dictionary containing the data to be transformed. The CoT prompt will be injected into the "prompt" key.
    :return: The input data with the CoT prompt injected.
    """

    def __init__(self, cot_prompt: str = "Your task is: {instruction}. Please think and generate the next action chunk based on the current observations."):
        super().__init__()
        self.cot_prompt = cot_prompt

    def __call__(self, data):
        if "prompt" in data:
            data["prompt"] = self.cot_prompt.format(instruction=data["prompt"])
        return data

