import pandas as pd
import sys, os

def convert(data_path, out_name, g2p_path="data/g2P/ulster.tsv",data_columns=None):
    """_summary_
    Converts 
    
    Args:
        g2p_path (string): path to g2p tsv file
        data_path (string): _description_
        out_name (string): _description_
        data_columns (list, optional): list of data columns, one of which should be "sentences" containing grapheme representation. Defaults to None.
    """
    transcriptions = pd.read_csv(data_path, names=data_columns)
    
    # turn df into dict for simple lookup
    g2p_dict = _getg2p()
    
    transcriptions["phoneme_sentence"] = transcriptions["sentence"].apply(_sent2phones)

def _getg2p(g2p_path="../../data/g2P/ulster.tsv"):
    g2p_file = pd.read_csv(g2p_path,sep="\t", names=["word","phonemes"])
    # turn df into dict for simple lookup
    g2p_dict = g2p_file.set_index("word")["phonemes"].to_dict()
    return g2p_dict
    

def _sent2phones(sentence, g2p_dict):
    """
    Helper method that translates a sentence to phoneme representation.

    Args:
        sentence (_type_): grapheme sentence
        g2p_dict (_type_): grapheme-to-phoneme dictionary to translate with

    Returns:
        sequence of words in phoneme form
    """
    words = [x.strip(" .,!?:;") for x in sentence.split()]
    
    phoneme_seq = []
    for word in words:
        if word in g2p_dict:
            phoneme_seq.append(g2p_dict[word].replace(" ",""))
        elif word.lower() in g2p_dict:
            phoneme_seq.append(g2p_dict[word.lower()].replace(" ",""))
        else:
            phoneme_seq.append("[UNK]")
    
    return {"phoneme_sentence": "|".join(phoneme_seq)}

#fleurs_train_path = r"../../data/fleurs/transcription/data%2Fga_ie%2FTrain.tsv"
#fleurs_dev_path = r"../../data/fleurs/transcription/data%2Fga_ie%2FDev.tsv"
#fleurs_test_path = r"../../data/fleurs/transcription/data%2Fga_ie%2FTest.tsv"

#output_name = "fleurs"

if __name__ == "__main__":
    if len(sys.argv) > 2:
        path = sys.argv[1]
        output_name = sys.argv[2]
    else:
        path = "data/fleurs/transcription/data%2Fga_ie%2FTrain.tsv" 
        path = "data/fleurs/transcription/data%2Fga_ie%2Ftrain.tsv"
        output_name = "fleurs"
    print(sys.path)
    convert(path,output_name,data_columns=["path","transcript"])