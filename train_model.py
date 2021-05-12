import os
import logging
import random
import itertools

import torch
import torch.optim as optim

from tqdm import tqdm

from Preprocess import PreprocessSentences
from WordEmbedding import CreateEmbeddings

from bilstm_crf import BiLSTM_CRF

logger = logging.getLogger(__name__)

def main():

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cude.manual_seed(1)

    ## Read training data
    trainfile = open('data/train/train.txt','r', encoding = 'utf8')

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

    print("finished reading file")

    ## Preprocess Data
    SentenceList, LabelsList = PreprocessSentences(SentenceList, LabelsList)

    print("finished Pre processing")

    ## Get Word Embeddings

    EmbeddingsList, LabelsList = CreateEmbeddings(SentenceList, LabelsList)

    print("finished Embedding")

    ## Create Model
    model = BiLSTM_CRF()

    ## Move model to GPU if running with Cuda
    #model = model.cuda()

    ## Create Optimizer
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr= .5)

    ## Train for epochs using tqdm
    for i in tqdm(range(5), unit = 'epoch'):
        train_epoch(model, EmbeddingsList, LabelsList, optimizer)

def train_epoch(model, EmbeddingsList, LabelsList, optimizer):

    max_size = len(EmbeddingsList[0])

    MaskList = []
    for Embedding in EmbeddingsList:
        mask = []
        for word in Embedding:
            mask.append(0 if (sum(word) == 0) else 1)
        MaskList.append(mask)

    temp = list(zip(EmbeddingsList, LabelsList, MaskList))
    random.shuffle(temp)
    EmbeddingsList, LabelsList, MaskList = zip(*temp)

    EmbeddingsList = torch.LongTensor(EmbeddingsList)
    LabelsList = torch.LongTensor(LabelsList)
    MaskList = torch.ByteTensor(MaskList)

    EmbeddingsList = torch.split(EmbeddingsList,5)
    LabelsList = torch.split(LabelsList,5)
    MaskList =torch.split(MaskList, 5)

    model.train()

    for batch, labels, masks in tqdm(itertools.zip_longest(EmbeddingsList,LabelsList, MaskList), total = len(EmbeddingsList)):

        emmissions = model(batch, masks, max_size)

        loss = -model.crf(emmissions,labels, mask = masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.global_step += 1

    ## Save Epoch



if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()