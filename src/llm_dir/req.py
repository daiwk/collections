#encoding=utf8
import requests
import json
prompt = "tell me 5 products similar to iphone 14 promax 256G\n"
prompt = "列举5个和iphone 14 promax 256G相似的商品"
prompt = "where are you from"
#prompt = "who are you"
r = requests.get("http://127.0.0.1:5000?prompt=%s" % prompt )
res = r.text
print(res)
js = json.loads(res)
print(js)
#print(js[0])
#print(js[0]["generated_text"])

