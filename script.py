import argparse
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import itertools
from transformers import AutoTokenizer

def load_platypus():
    data = load_dataset(
        "garage-bAInd/Open-Platypus",
        split="train",
        cache_dir="/home/jovyan/quantization-vol-1/.cache/huggingface/datasets"
    )

    def concatenate_data(x):
        if x["input"] == None:
            return {"text": x["instruction"] + '\n' + x["output"]}
        else:
            return {"text": x["input"] + '\n' + x["instruction"] + '\n' + x["output"]}
        
    concat = data.map(concatenate_data)
    return [text for text in concat["text"]]

def load_capybara():
    data = load_dataset(
        "LDJnr/Capybara",
        split="train",
        cache_dir="/home/jovyan/quantization-vol-1/.cache/huggingface/datasets"
    )
    flat_list = []
    for example in data:
        for turn in example["conversation"]:
            inp = turn.get("input", "")
            out = turn.get("output", "")
            # Concatenate input and output, or just output if input is empty
            if inp:
                flat_list.append(inp + '\n' + out)
            else:
                flat_list.append(out)
    return flat_list

def load_chatqa():
    data = load_dataset(
        "nvidia/ChatQA-Training-Data",
        split="train",
        cache_dir="/home/jovyan/quantization-vol-1/.cache/huggingface/datasets"
    )

    def concatenate_data(x):
        content = x["messages"]["content"]
        document = x["document"]
        answers = x["answers"]
        return {"text": document + '\n' + content + '\n' + answers}
    
    concat = data.map(concatenate_data)
    return  [text for text in concat["text"]]

def create_calibration_dataset(max_calib_samples, max_calib_seq_len):
    platypus_data = load_platypus()
    capybara_data = load_capybara()
    chatqa_data = load_chatqa()

    # Extract text fields from each dataset
    platypus_texts = [x["text"] for x in platypus_data]
    capybara_texts = [x["text"] for x in capybara_data]
    chatqa_texts = [x["text"] for x in chatqa_data]

    datasets = [platypus_texts, capybara_texts, chatqa_texts]

    def truncate(text):
        return text[:max_calib_seq_len]

    # Round-robin sampling across all datasets
    interleaved = itertools.zip_longest(*datasets, fillvalue=None)

    samples = []
    for group in interleaved:
        for text in group:
            if text is not None:
                samples.append({"text": truncate(text)})
            if len(samples) >= max_calib_samples:
                return samples

    return samples
def main():
    parser = argparse.ArgumentParser(description="CLI for model quantization and saving")
    parser.add_argument("--hf_model_path", type=str, required=True, help="Path to the Hugging Face model")
    parser.add_argument("--quant_name", type=str, required=True, help="Name of the quantized model")
    parser.add_argument("--local_save_path", type=str, required=True, help="Path to save the quantized model")

    # Quantization config arguments
    parser.add_argument("--zero_point", action="store_true", help="Enable zero point for quantization")
    parser.add_argument("--no-zero_point", action="store_false", dest="zero_point", help="Disable zero point for quantization")
    parser.add_argument("--q_group_size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--w_bit", type=int, default=4, help="Weight bit width")
    parser.add_argument("--version", type=str, default="GEMM", help="Quantization version")

    # Model config arguments
    parser.add_argument("--device_map", type=str, default=None, help="Device map for loading the pretrained model")

    # Quantize parameters
    parser.add_argument("--max_calib_samples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument("--max_calib_seq_len", type=int, default=512, help="Calibration sample sequence length.")

    args = parser.parse_args()

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }

    print(f"Loading model from: {args.hf_model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        args.hf_model_path,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)

    print(f"Quantizing model with config: {quant_config}")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.max_calib_seq_len,
    )

    print(f"Saving quantized model to: {args.local_save_path}")
    model.save_quantized(args.local_save_path)
    tokenizer.save_pretrained(args.local_save_path)

    print(f"Quantized model '{args.quant_name}' saved successfully.")

if __name__ == "__main__":
    main()