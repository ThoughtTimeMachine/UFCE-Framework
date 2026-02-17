import os
from sentence_transformers import SentenceTransformer

# Define where we want the model to live permanently
model_path = os.path.join(os.getcwd(), "models", "nomic_embed")

print(f"Downloading model to: {model_path}...")
os.makedirs(model_path, exist_ok=True)

# Download and save specifically to this folder
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
model.save(model_path)

print("SUCCESS: Model saved locally.")
print("You can now disconnect the internet.")