import re
import os
import unicodedata
from typing import Iterable, Union, Literal
from collections import deque

      
# txt_file= "Wikipedia scraping data/Red panda/Red panda.txt"

txt_data= "ä, ö, ü and ß  \u200D  sjsj!!!JHIgu___jss"

# with open(txt_file, "r") as f :
#     txt_data= f.read()
    
# print(txt_data)

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text) # Joins accents
    return text

def __remove_accents(text, en_only=True):
    for letter in text:
        if en_only and letter in "أآإؤئ":
            normal = letter
        else:
            normal = unicodedata.normalize("NFKD", letter)
        if len(normal) > 1:
            for n in normal:
                if n.isalpha():
                    yield n
        else:
            yield normal
            
def normalize_accents(text: str, mode: Union[Literal["separate"], Literal["remove"]] = "remove"):
    if mode == "remove":
        return ''.join(__remove_accents(text))
    if mode == "separate":
        return unicodedata.normalize("NFKD", text)
    raise NotImplementedError(f"Mode {mode} can only be `separate` or `remove`")


def clean_text(text):
        
    cleaned_text= normalize_unicode(text)
    cleaned_text= normalize_accents(cleaned_text)
    
    cleaned_text= cleaned_text.lower()
    cleaned_text= re.sub(r'[^a-z0-9\s\n]','',cleaned_text)
    cleaned_text= re.sub(r'[\t\r]+',' ',cleaned_text)
    cleaned_text= re.sub(r'\n+','\n',cleaned_text)
   

    # print(cleaned_text)
    
    

    
    
    return cleaned_text

# print(clean_text(txt_data))
    

      
      
      
if __name__ == "__main__":
    
#For all txt files

    # for root, dirs, files in os.walk(".", topdown=False):
        # for name in files:
        #     print(os.path.join(root, name))
        

    for root, dirs, files in os.walk("./Wikipedia scraping data copy"):
        # os.makedirs('./Wikipedia scraping data clean/{dir}')
        # for dir in dirs:
        #     os.makedirs(f'./Wikipedia scraping data clean/{dir}')
            
        for file in files:
            # print(file)
            #   print(os.path.join(root, file))
            # print(file)
            if file.endswith(".txt"):
                # print(file)
                with open(os.path.join(root, file), "r", encoding="utf-8" ) as f:
                    file_text= f.read()
            
                cleaned_text= clean_text(file_text)
                dir= root.split("/")[-1] 
                # print(dir)
                path= os.path.join("./Wikipedia scraping data clean", dir)
                print(path)
                    
                with open(os.path.join(path, file), "w", encoding="utf-8" ) as f:
                    f.write(cleaned_text)
                
                    # print(cleaned_text)
                    # f.write(clean_text())
          
      
    
    # with open('./Wikipedia scraping data copy/Uyghurs/Culture.txt', "r", encoding="utf-8" ) as f:
    #     content= f.read()
  
    # content= clean_text(content)  
    # with open('./Wikipedia scraping data clean/Culture.txt', "w", encoding="utf-8" ) as f:
    #     f.write(content)
    # print(repr(content))


# Print lines to show that newlines are preserved
    # for line in lines:
    #     # print(repr(line))
    #     cleaned_text=clean_text(line)
    
    
    # with open('./History.txt', "w", encoding="utf-8" ) as f:
    #     f.write(cleaned_text)
        
        # f.seek(0)
        # f.write(cleaned_text)
        # f.truncate()
        
    #                 file_text= f.read()
    #                 cleaned_text= clean_text(file_text)
                    
    #                 print(cleaned_text)
    #                 # f.write(clean_text())
      



