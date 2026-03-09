#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import time
from tqdm import tqdm
import re
import subprocess

from carbontracker.tracker import CarbonTracker

# Fill this in
CARBON_API_KEY = 
OPENAI_API_KEY = 
AI21_API_KEY = 

def main(model, name, file):
    df = pd.read_csv(file, encoding = 'cp1252', sep=';', header=0)
    #selects model to use for the translation of the chosen file
    if model=='mbart':
        mbart(df, name)
    elif model=='j2':
        j2(df, name)
    elif model=='gpt':
        gpt(df, name)
    elif model=='deepseek':
        deepseek(df, name)
    elif model=='mistral':
        mistral(df, name)
    elif model=='tower':
        tower(df, name)

def mbart(df, name):
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

    translated = []

    tracker = CarbonTracker(epochs=9, components="gpu",
              api_keys={"electricitymaps": CARBON_API_KEY},
              log_dir="carbon_log_mbart", verbose=0)
    for line in tqdm(df['EN']):
        tracker.epoch_start()
        model_inputs = tokenizer(line, return_tensors="pt")

        # translate from English to Dutch
        generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["nl_XX"]
        )
        tr = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        tracker.epoch_end()
        translated.append(tr[0])

    tracker.stop()
    translated = pd.DataFrame(translated, columns = ['translation'])    
    translated.to_csv("mbart_"+name+'.out.csv', index=False, sep='\t', encoding='utf-8')  

def j2(df, name):
    from ai21 import AI21Client
    from ai21.models.chat import ChatMessage
    client = AI21Client(api_key=AI21_API_KEY)

    translated = []
    for line in tqdm(df['EN']):
        prompt_x = [ChatMessage(content = "Translate the following text from English to Dutch, showing only the translation: "+ line, role="user")]
        response = client.chat.completions.create(
            model="jamba-1.5-large",
            messages = prompt_x,
            numResults=1,
            maxTokens=300,
            temperature=0,
            topKReturn=0,
            topP=1,
#             stop=["\n"],
            countPenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
            },
            frequencyPenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
            },
            presencePenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
            },
            stopSequences=[]
        )
        txt_trans = response.choices[0].message.content
        #print(txt_trans)
        translated.append(txt_trans)
        time.sleep(3)
        
    translated = pd.DataFrame(translated, columns = ['translation'])    
    translated.to_csv("j2_"+name+'.out.csv', index=False, sep='\t', encoding='utf-8') 

def gpt(df, name):
    import openai

    openai.api_key = OPENAI_API_KEY
    translated = []

    for line in tqdm(df['EN']):
        prompt_x = "Show only translation, English: "+ line +" = Dutch:"

        # response = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt= prompt_x,
        #     temperature=0,
        #     max_tokens=256,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0
        # )
        response = openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_x,
                }
            ],
            #model="gpt-3.5-turbo-0125",
            #model="gpt-4o",
            model="gpt-4o-mini",
        )
        txt_trans = response.choices[0].message.content
        translated.append(txt_trans)
        time.sleep(2)
    translated = pd.DataFrame(translated, columns = ['translation'])    
    translated.to_csv("gpt_"+name+'.out.csv', index=False, sep='\t', encoding='utf-8')   

def deepseek(df, name):
    #import subprocess

    #subprocess.run(["ollama", "serve"])

    import ollama

    translated = []

    tracker = CarbonTracker(epochs=len(df['EN']), components="gpu", 
              api_keys={"electricitymaps": CARBON_API_KEY}, 
              log_dir="carbon_log_deepseek", verbose=0)
    for line in tqdm(df['EN']):
        tracker.epoch_start()
        prompt_x = "Translate the following text from English to Dutch while preserving the original meaning, tone, and context. Maintain proper grammar and natural fluency as if written by a native speaker. Don't show the source nor your reasoning. Only show the final translation. The text is: "+line
        response = ollama.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt_x,
                }
            ],
            model="deepseek-r1:70b",
            options={"temperature":1.3}
        )
        response_content = response["message"]["content"]
        tracker.epoch_end()
        txt_trans = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
        txt_trans = re.sub(r"\n", " ", txt_trans, flags=re.DOTALL)
        translated.append(txt_trans)

    tracker.stop()
    translated = pd.DataFrame(translated, columns = ['translation'])
    translated.to_csv("deepseek_"+name+'.out.csv', index=False, sep='\t', encoding='utf-8')  

def mistral(df, name):
    #import subprocess

    #subprocess.run(["ollama", "serve"])

    import ollama

    translated = []

    tracker = CarbonTracker(epochs=len(df['EN']), components="gpu", 
              api_keys={"electricitymaps": CARBON_API_KEY}, 
              log_dir="carbon_log_mistral", verbose=0)

    for line in tqdm(df['EN']):
        tracker.epoch_start()
        prompt_x = "Show only translation, English: "+ line +" = Dutch:"

        response = ollama.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt_x,
                }
            ],
            model="mistral-large",
            options={"temperature":0.2}
        )
        response_content = response["message"]["content"]
        tracker.epoch_end()
        txt_trans = re.sub(r"\n", " ", response_content.rstrip().lstrip(), flags=re.DOTALL)
        translated.append(txt_trans)

    tracker.stop()
    translated = pd.DataFrame(translated, columns = ['translation'])
    translated.to_csv("mistral_"+name+'.out.csv', index=False, sep='\t', encoding='utf-8')  

def tower(df, name):
    #import subprocess

    #subprocess.run(["ollama", "serve"])

    import ollama

    translated = []

    tracker = CarbonTracker(epochs=len(df['EN']), components="gpu", 
              api_keys={"electricitymaps": CARBON_API_KEY}, 
              log_dir="carbon_log_tower", verbose=0)
    for line in tqdm(df['EN']):
        tracker.epoch_start()
        prompt_x = "Translate the following text from English into Dutch.\nEnglish: " + line + "\nDutch:"

        response = ollama.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt_x,
                }
            ],
            model="thinkverse/towerinstruct",
            options={"temperature":0.2}
        )
        response_content = response["message"]["content"]
        tracker.epoch_end()
        txt_trans = re.sub(r"\n", " ", response_content.rstrip().lstrip(), flags=re.DOTALL)
        translated.append(txt_trans)

    tracker.stop()
    translated = pd.DataFrame(translated, columns = ['translation'])
    translated.to_csv("tower_"+name+'.out.csv', index=False, sep='\t', encoding='utf-8')  

#for model in ["mbart", "j2", "gpt", "deepseek"]:
#for model in ["mbart", "deepseek", "mistral", "tower"]:

# Start the Ollama server in the background
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#for model in ["mbart", "deepseek", "mistral", "tower"]:
for model in ["mistral", "tower"]:
    for filename in ["literature", "news", "poetry"]:
        print(model + " " + filename)
        main(model, filename, os.path.join('data', filename+'-1000.csv'))
