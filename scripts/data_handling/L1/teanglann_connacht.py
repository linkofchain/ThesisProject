# imports
import os, requests, time
import pandas as pd
from tqdm import tqdm

# define output folder
script_dir = os.path.dirname(__file__)
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
output_dir = proj_root+"/data/teanglann/audio"

# grab g2p file for word list to scrape
g2p_path = proj_root+"/data/g2P/ulster.tsv"
g2p = pd.read_csv(g2p_path, sep="\t",names=["word","phonemes"])

def g2p2wordlist(g2pdf):
    words = []
    for row in g2pdf.iterrows():
        word = row[1].loc["word"]
        words.append(word)
    return words

words = g2p2wordlist(g2p)    

# scrape words
Ulster_URL_root = "https://www.teanglann.ie/CanU/"
htmlspace = "%20"

def get_teanglann_audio(wordlist,checkpnt):
    prog_bar = tqdm(wordlist[wordlist.index(checkpnt):],
                    initial=wordlist.index(checkpnt),
                    total=len(wordlist))
    
    for word in prog_bar:
        prog_bar.set_description("Scraping: " + word)
        url = f"{Ulster_URL_root}{word}.mp3"
        outfile = f"{output_dir}/{word}.mp3"
        
        response = requests.get(url)
        if response.status_code == 200: # 200 is okay, 404 not found
            with open(outfile, "wb") as f: # wb for nontext data
                f.write(response.content)
            with open("scripts/datahandling/L1/teanglann.checkpoint.txt", "w") as chk:
                chk.write(word)  
                
        else:
            prog_bar.set_description(f"{word} not found. Skipping...")
            
        time.sleep(2)



checkfile = open("scripts/datahandling/L1/teanglann.checkpoint.txt", "r")
checkpoint = checkfile.read()
checkfile.close()   
get_teanglann_audio(words,checkpoint)
# https://pypi.org/project/tqdm/ progress bar!