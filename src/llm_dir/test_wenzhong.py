from transformers import GPT2Tokenizer,GPT2LMHeadModel
hf_model_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)
question = "北京是中国的"
inputs = tokenizer(question,return_tensors='pt')
generation_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_length=150,
                                # max_new_tokens=80,
                                do_sample=True,
                                top_p = 0.6,
                                # num_beams=5,
                                eos_token_id=50256,
                                pad_token_id=0,
                                num_return_sequences = 5)

for idx,sentence in enumerate(generation_output.sequences):
    print('next sentence %d:\n'%idx,
    tokenizer.decode(sentence).split('<|endoftext|>')[0])
    print('*'*40)
