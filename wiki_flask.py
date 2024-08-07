from flask import Flask, request, jsonify
from pymongo import MongoClient
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import os
import pprint
import json
from bson import ObjectId

import nltk
nltk.download('punkt')

db_user= os.environ.get("DB_USER")
db_pass= os.environ.get("DB_PASS")

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

    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=20)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
    
    response= {"title": predicted_title}
    return response
     
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