import torch
import pickle
import gzip
import os

def test_model_file(filepath):
    """Test loading a model file"""
    try:
        print(f"Testing: {filepath}")
        
        if filepath.endswith('.pkl'):
            # Try regular pickle first
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Loaded with pickle: {type(data)}")
                return data, "pickle"
            except:
                # Try gzip pickle
                try:
                    with gzip.open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    print(f"✅ Loaded with gzip pickle: {type(data)}")
                    return data, "gzip_pickle"
                except Exception as e:
                    print(f"❌ Failed to load: {e}")
                    return None, None
        
        elif filepath.endswith('.pth'):
            data = torch.load(filepath, map_location='cpu')
            print(f"✅ Loaded torch file: {type(data)}")
            return data, "torch"
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# Test the complete model files
model_files = [
    "models/final_trained_urdu_roman_nmt_WITH_TOKENIZER_PICKLES.pkl",
    "models/best_urdu_roman_nmt_model_WITH_TOKENIZER_PICKLES.pkl",
    "models/nmt_model.pth"
]

working_file = None
for file in model_files:
    if os.path.exists(file):
        data, method = test_model_file(file)
        if data is not None:
            working_file = (file, data, method)
            print(f"🎉 Found working file: {file}")
            
            # Check what's inside
            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
                if 'model' in data:
                    print(f"Model type: {type(data['model'])}")
                if 'urdu_tokenizer' in data:
                    print("✅ Has Urdu tokenizer")
                if 'roman_tokenizer' in data:
                    print("✅ Has Roman tokenizer")
            break

if working_file:
    print(f"\n🚀 Use this file for the Streamlit app: {working_file[0]}")
else:
    print("\n❌ No working model files found")