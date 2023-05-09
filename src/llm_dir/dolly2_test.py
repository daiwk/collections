#encoding=utf8
## sudo pip3 uninstall nvidia_cublas_cu11
import sys
import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
num = int(sys.argv[1])
product = sys.argv[2]
prompt = "列举%d个和%s相似的商品" % (num, product)
print(prompt)
print(generate_text(prompt))

