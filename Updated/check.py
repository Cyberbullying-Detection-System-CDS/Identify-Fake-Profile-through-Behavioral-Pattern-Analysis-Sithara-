import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'fake_profile_model_new.pkl')
LE_PATH    = os.path.join(SCRIPT_DIR, 'label_encoder_new.pkl')

print(f"Looking for model at:   {MODEL_PATH}")
print(f"Looking for encoder at: {LE_PATH}")
print(f"Model exists:   {os.path.exists(MODEL_PATH)}")
print(f"Encoder exists: {os.path.exists(LE_PATH)}")

# List all files in the folder
print("\nFiles in folder:")
for f in os.listdir(SCRIPT_DIR):
    print(f"  {f}")