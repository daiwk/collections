#encoding=utf8
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

#generator = pipeline(model="nlpcloud/instruct-gpt-j-fp16", torch_dtype=torch.float16, device=0)

model_name = "bigscience/bloom"
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             torch_dtype="auto", 
                                             device_map="auto")

from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def get_res():
   if request.method == 'POST':
      print(1)
      user = request.form['nm']
      return "x"
   else:
      print(2)
      prompt = request.args.get('prompt')
      res = model.generate(prompt)
      return res

if __name__ == '__main__':
    app.run()

