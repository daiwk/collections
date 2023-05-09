#encoding=utf8
#sudo pip uninstall nvidia_cublas_cu11
#cpm_kernels
#transformers==4.28.1
#gradio
#mdtex2html
#sentencepiece
#accelerate
# https://github.com/THUDM/ChatGLM-6B

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
iters = 100
for i in range(0, iters):
    try:
        response, history = model.chat(tokenizer, input("You: "), history=history)
    except Exception as e:
        print(str(e))
        continue
    print("Bot:", response)

