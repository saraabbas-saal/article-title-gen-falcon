from pathlib import Path
import os
import numpy as np
import pandas as pd
import json


def read_data(dir):
    texts = []
    labels = []

    for root, dirs, files in os.walk("./Wikipedia scraping data clean"):
        # print(dirs)
        for file in files:
            if file.endswith(".txt"):
                filename=  file.split(".")[0]
                print("***********\nfilename:",filename)
                filepath= os.path.join(root, file)
                # print(filepath)
                foldername= root.split("/")[-1]
                # print("foldername:",foldername)
                if file.startswith(foldername):
                    label= filename
                else:
                    label= foldername+" "+filename
                    
                with open(filepath, "r", encoding="utf-8" ) as f:
                    file_text= f.read()
                    texts.append(file_text)
                    # print(label)
                    
                    labels.append(label)

    data= {'texts':texts, 'titles':labels}
    data = [{"content": c, "title": t} for c, t in zip(texts, labels)]

# Save the list of dictionaries as a JSON file
    with open("wiki_data.json", "w") as f:
        json.dump(data, f, indent=4)

# Display the JSON data
    print(json.dumps(data, indent=4))
    
    # data= pd.DataFrame(data)
    # data.to_csv("wiki_data.csv", index=False)
    
    
    return texts,labels
                

texts, labels = read_data('./Wikipedia scraping data clean')

# # Assume you have your data as lists
# # texts = ["text1", "text2", "text3", ...]  # List of texts
# # labels = ["title1", "title2", "title3", ...]  # Corresponding list of titles

# # Create a pandas DataFrame
# data = pd.DataFrame({'text': texts, 'title': labels})
# # Install necessary libraries


# # Assume you have your data as lists
# texts = ["text1", "text2", "text3", ...]  # List of texts
# labels = ["title1", "title2", "title3", ...]  # Corresponding list of titles

# # Create a pandas DataFrame
# data = pd.DataFrame({'text': texts, 'title': labels})

# # Convert to Hugging Face dataset
# dataset = Dataset.from_pandas(data)

# # Load pre-trained model and tokenizer
# tokenizer = LlamaTokenizer.from_pretrained('huggingface/llama')
# model = LlamaForCausalLM.from_pretrained('huggingface/llama')

# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Prepare for training
# tokenized_dataset = tokenized_dataset.rename_column("title", "labels")
# tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset,
# )

# # Train the model
# trainer.train()

# # Save the finetuned model
# model.save_pretrained("finetuned_llama_title_generator")
# tokenizer.save_pretrained("finetuned_llama_title_generator")

# # Function to generate titles
# def generate_title(text):
#     inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     outputs = model.generate(**inputs, max_length=20)
#     title = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return title

# # Example usage
# text = "Your custom text here"
# print(generate_title(text))
