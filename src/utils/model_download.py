from transformers import AutoTokenizer, AutoModelForCausalLM

# Llama 3 8B Instruct
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=r"C:\Users\TEC-INT02\Documents\Agent-IA\models\base")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=r"C:\Users\TEC-INT02\Documents\Agent-IA\models\base")
