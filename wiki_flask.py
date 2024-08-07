from flask import Flask, request, jsonify
from pymongo import MongoClient
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import os
import requests
import json
# from bson import ObjectId
from ai71 import AI71 
import nltk
nltk.download('punkt')

db_user= os.environ.get("DB_USER")
db_pass= os.environ.get("DB_PASS")
api_key= os.environ.get("AI71_API_KEY")

app= Flask("wiki_app")


def write_db():
    # references={}
    for root, dirs, files in os.walk("./Wikipedia scraping data clean"):
        # print(dirs)
        for file in files:
                filepath= os.path.join(root, file)
                # print(filepath)
                foldername= root.split("/")[-1]
                # print("foldername:",foldername)
                if file.endswith(".json"):
                    
                    with open(filepath, "r", encoding="utf-8" ) as f:
                        reference= json.load(f)
                        if reference:
                        # print(reference)
                            # references[foldername]= reference
                            collection.insert_one({foldername.lower(): reference})
    
    
    # collection.insert_many(references)
    # pprint.pp(references)
    
       
    
def load_model(model_dir):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model

def load_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return tokenizer

def serialize_document(doc):
    if doc is None:
        return None
    doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
    return doc

model_dir = "checkpoint-400"
model = load_model(model_dir)
tokenizer= load_tokenizer(model_dir)
    

        
@app.route("/topic/generate",methods=['POST'])
def generate_topic():
    
    max_input_length = 512
    text= request.json['text']
    
    inputs = ["Generate a short title: " + text]
    
client = AI71(api_key)
messages = [{"role": "system", "content": "You work at wikipedia. Generate a title for the article provided by the user input."}]

class AI71:
        def __init__(self, api_key):
            self.api_key = api_key
            
        def chat_completions(self, model, messages, temperature=0.7):
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            }
            json_data = {
                'model': model, #'tiiuae/falcon-180B-chat',
                'messages': messages,
                'temperature':temperature
            }
            response = requests.post('https://api.ai71.ai/v1/chat/completions', headers=headers, json=json_data)
            return response.json()


content = input(f"User:")
messages.append({"role": "user", "content": content})
print(f"Falcon:", sep="", end="", flush=True)
content = ""

# for chunk in client.chat.completions.create(
#     messages=messages,
#     model="tiiuae/falcon-180B-chat",
#     stream=True,
# ):
#     delta_content = chunk.choices[0].delta.content
#     if delta_content:
#         print(delta_content, sep="", end="", flush=True)
#         content += delta_content

# messages.append({"role": "title generator", "content": content})
# print("\n")


client = AI71("apikey")
message = [{"role": "system", "content": "You work at wikipedia. Generate a title for the article provided by the user input."},{"role": "user", "content": "Hello!"}]
chunk = client.chat_completions('tiiuae/falcon-180B-chat', message)
print(chunk)
output = chunk["choices"][0]["message"]["content"]

    # inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    # output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=20)
    # decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
    
    # response= {"title": predicted_title}
    # return response
     
@app.route("/references",methods=['GET'])
def fetch_ref():
    
    topic= request.args.get('topic')
    print(topic)
    topic= topic.lower()
    # collection.find_one(topic)
    # document = collection.find_one({topic: {"$exists": True}})
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    

    document = collection.find_one({topic: {"$exists": True}})
    
    if document:
        # print(document)
        document = serialize_document(document) 
        return document
    else:
        return jsonify({"error": "No document found with the specified key"}), 404

    # print(document)
    # return jsonify(reference= document )
  
    
if __name__ == "__main__":
    client= MongoClient(
        f"mongodb+srv://{db_user}:{db_pass}@cluster0.dl2jeop.mongodb.net/" )
    db= client.get_database('wikiProject')
    collection= db.referencez
    write_db()
    
    app.run(debug=True, host= "0.0.0.0", port=5000)