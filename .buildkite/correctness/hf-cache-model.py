from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_url)
    model = AutoModelForCausalLM.from_pretrained(args.model_url)

    tokenizer.save_pretrained("hf-cache")
    model.save_pretrained("hf-cache")