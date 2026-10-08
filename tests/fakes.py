from core.contract import LLMResponse

class FakeChatModel:
    """离线替身，满足 ChatModel 契约，零网络。"""
    def __init__(self, responder, provider: str = "fake", model_name: str = "fake-1"):
        self.provider = provider
        self.model_name = model_name
        self._responder = responder

    def chat(self, prompt, system_prompt=None, json_model=False,
             temperature=0.0, max_tokens=None) -> LLMResponse:
        text = self._responder(prompt)
        return LLMResponse(text=text, prompt_tokens=len(prompt) // 4,
                           completion_tokens=len(text) // 4, latency_s=0.0)

    def generate(self, prompt, system_prompt=None, json_model=False,
                 temperature=0.0, seed=None) -> str:
        return self.chat(prompt, system_prompt, json_model, temperature).text