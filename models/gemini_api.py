# setup
import google.generativeai as genai
 
 
genai.configure(api_key='key')  
 
# 查询模型
for m in genai.list_models():
    print(m.name)
    print(m.supported_generation_methods)


model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("告诉我太阳系中最大行星的相关知识")
print(response.text)
