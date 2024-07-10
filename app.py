import gradio as gr
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

def text_generator(INPUT, WORDS):
    input_ids = tokenizer.encode(INPUT, return_tensors='tf')
    pre_out = model.generate(
        input_ids, max_length=WORDS, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(
        pre_out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

demo = gr.Interface(
    fn=text_generator, 
    inputs=["text", gr.Slider(
        value=150,
        minimum=50,
        maximum=500,
        step=1
    )],
    outputs=[gr.Textbox(
        label="OUTPUT", lines=3
    )],
    title="TEXT_GENERATIVE_AI",
    description="GPT-2 is a transformers model pretrained \
            on a very large corpus of English data in a \
            self-supervised fashion. This model was created using \
            'hugging face transformers', which uses pytorch and tensorflow. \
            GPT-2 works like a traditional language model is that it takes \
            word vectors and input and produces estimates for the probability \
            of the next word as outputs. It is auto-regressive in nature: \
            each token in the sentence has the context of the previous words. \
            Thus GPT-2 works one token at a time. NOTE: This model is best for \
            answering simple questions that are often repetitive in customer \
            support.",
    article="Time usually taken: 20 s" 
)

if __name__ == "__main__":
    demo.launch()