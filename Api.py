import google.generativeai as genai

genai.configure(api_key="AIzaSyBvblnLVgJ50eZyGzaEqx-tMC94lZNRSHk")

models = genai.list_models()

for model in models:
    print(f"Model Name: {model.name}")
    print(f"Display Name: {model.display_name}")
    print(f"Description: {model.description}")
    print(f"Supported Methods: {model.supported_generation_methods}")
    print(f"Input Token Limit: {model.input_token_limit}")
    print(f"Output Token Limit: {model.output_token_limit}")
    print(f"Temperature: {model.temperature}")
    print(f"Top K: {model.top_k}")
    print(f"Top P: {model.top_p}")
    print("-" * 50)