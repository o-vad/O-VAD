import torch
# from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers import AutoProcessor, AutoModelForImageTextToText


class QwenSingleton:
    _model = None
    _processor = None
    # _model_id = "/projects/bgiv/qilong/Qwen3-VL-30B-A3B-Instruct"
    _model_id = "/projects/bgiv/qilong/Qwen3-VL-32B-Instruct"

    @classmethod
    def get_model_and_processor(cls):
        """Loads the model once and returns the cached instances."""
        if cls._model is None or cls._processor is None:
            print("🚀 Loading Qwen3-VL into GPU for the first time... (This only happens once!)")
            cls._processor = AutoProcessor.from_pretrained(cls._model_id)
            # cls._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            #     cls._model_id,
            #     torch_dtype=torch.bfloat16,
            #     device_map="auto"
            # )
            cls._model = AutoModelForImageTextToText.from_pretrained(
                cls._model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            cls._model.eval()
        else:
            # Optional: Add a debug print to verify it's reusing the model
            # print("⚡ Reusing cached Qwen3-VL model from GPU.")
            pass
            
        return cls._model, cls._processor
    