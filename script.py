import argparse
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import itertools
from transformers import AutoTokenizer

def load_platypus():
    """
    Loads the Open-Platypus dataset from HuggingFace's dataset hub.
    
    The dataset is concatenated such that the instruction is always the second line,
    and the output is always the third line. If the input is None, it is dropped.
    
    Args:
        None
    
    Returns:
        A list of concatenated strings.
    """
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
    
    """
    Loads the Capybara dataset and concatenates the input and output of each turn in the conversation
    into a single string, separated by a newline.

    Returns:
        A list of strings, where each string is a concatenated input and output of each turn in the conversation.
    """
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
    """
    Loads the ChatQA dataset and concatenates the document, the content of the messages and the answers
    into a single string, separated by newlines.

    Returns:
        A list of strings, where each string is a concatenated document, content and answer.
    """
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

def create_calibration_dataset(max_calib_samples=128, max_calib_seq_len=512):
    """
    Combines and truncates samples from Platypus, Capybara, and ChatQA datasets
    to be used for calibration during quantization.

    Args:
        max_calib_samples (int): Total number of samples to return.
        max_calib_seq_len (int): Maximum length of each text sample.

    Returns:
        A list of dicts with key "text", truncated to the specified length.
    """
    platypus_data = load_platypus()
    capybara_data = load_capybara()
    chatqa_data = load_chatqa()

    datasets = [platypus_data, capybara_data, chatqa_data]

    def truncate(text):
        return text[:max_calib_seq_len]

    # Interleave data from all datasets in round-robin fashion
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
    """
    CLI for model quantization and saving.

    This script takes a Hugging Face model path, a name for the quantized model, and a local save path
    as required arguments. It also accepts optional arguments for quantization and model configuration.

    The script will load the model from the given Hugging Face path, quantize it with the given
    configuration, and save the quantized model to the given local path.

    Args:
        pretrained_dir (str): Path to the pretrained model
        quant_dir (str): Path to save the quantized model

    Optional Args:
        zero_point (bool): Enable zero point for quantization (default: True)
        q_group_size (int): Quantization group size (default: 128)
        w_bit (int): Weight bit width (default: 4)
        version (str): Quantization version (default: "GEMM")
        device_map (str): Device map for loading the pretrained model (default: None)
        max_calib_samples (int): Number of calibration samples (default: 128)
        max_calib_seq_len (int): Calibration sample sequence length (default: 512)
    """
    parser = argparse.ArgumentParser(description="CLI for model quantization and saving")
    parser.add_argument("--pretrained_dir", type=str, required=True, help="Path to the Hugging Face model")
    parser.add_argument("--quant_dir", type=str, default=None, help="Path to save the quantized model")

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
    if args.quant_dir is None:
        if args.pretrained_dir.endswith("/"):
            args.quant_dir = f"{args.pretrained_dir[:-1]}-{args.w_bit}bit-awq"
        else:
            args.quant_dir = f"{args.pretrained_dir}-{args.w_bit}bit-awq"

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }

    print(f"Loading model from: {args.pretrained_dir}")
    model = AutoAWQForCausalLM.from_pretrained(
        args.pretrained_dir,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir, trust_remote_code=True)

    print(f"Quantizing model with config: {quant_config}")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.max_calib_seq_len,
        calib_data=create_calibration_dataset(args.max_calib_samples, args.max_calib_seq_len),
    )

    print(f"Saving quantized model to: {args.quant_dir}")
    model.save_quantized(args.quant_dir)
    tokenizer.save_pretrained(args.quant_dir)

    quant_name = args.quant_dir.split("/")[-1]
    print(f"Quantized model {quant_name} saved successfully.")

if __name__ == "__main__":
    main()