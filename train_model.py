import os
import logging

import torch

from tqdm import tqdm

from Preprocess import PreprocessSentences
from WordEmbedding import CreateEmbeddings

logger = logging.getLogger(__name__)

def main():

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cude.manual_seed(1)

    ## Read training data
    trainfile = open('data/tiny_train/tinytrain.txt','r', encoding = 'utf8')

    SentenceList = []
    LabelsList = []

    currentSentence = []
    currentLabels = []

    while True:
        line = trainfile.readline()

        if not line:
            SentenceList.append(currentSentence)
            LabelsList.append(currentLabels)
            break

        if line != '\n':
            ## Add to current
            line = line.rstrip()
            res = line.split('\t')

            currentSentence.append(res[0])
            currentLabels.append(res[1])
        else:
            ## Add to list
            SentenceList.append(currentSentence)
            LabelsList.append(currentLabels)

            currentSentence = []
            currentLabels = []

    trainfile.close()

    ## Preprocess Data
    SentenceList, LabelsList = PreprocessSentences(SentenceList, LabelsList)

    ## Get Word Embeddings

    EmbeddingsList, LabelsList = CreateEmbeddings(SentenceList, LabelsList)

    ## Create Model

    ## Move model to GPU if running with Cuda

    ## Create Optimizer

    ## Train for epochs using tqdm

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()