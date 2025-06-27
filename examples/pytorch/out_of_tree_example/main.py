import modeling_opt  # noqa
import modeling_phi3  # noqa

from tensorrt_llm import LLM


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    llm = LLM(model='microsoft/Phi-3-small-8k-instruct')
    outputs = llm.generate(prompts)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
