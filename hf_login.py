import argparse
from huggingface_hub import login

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Login")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face access token")
    args = parser.parse_args()
    
    login(token=args.token, add_to_git_credential=True)
    print("Successfully logged in!")
