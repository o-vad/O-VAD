# Using Claude (you're already using Claude, so you have access)
# export ANTHROPIC_API_KEY="your-key"
# python generate_mask_grounded.py --image frame.jpg --vlm claude --auto

# Or interactively select which objects to segment
export OPENAI_API_KEY="***REMOVED-CREDENTIAL***"
python annotate/generate_mask_grounded.py --image /home/yizhou/Mprojects/VAD/TubeletGraph/assets/example/01/000.jpg --vlm openai

# Using free local LLaVA (no API key needed)
# ollama serve  # Start Ollama
# ollama pull llava  # Download LLaVA model
# python generate_mask_grounded.py --image frame.jpg --vlm ollama --auto