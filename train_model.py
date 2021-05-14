import os
import logging
import random
import itertools

import torch
import torch.optim as optim

from tqdm import tqdm

from Preprocess import PreprocessSentences, PreprocessSentencesT
from WordEmbedding import CreateEmbeddings, CreateEmbeddingsT

from bilstm_crf import BiLSTM_CRF

logger = logging.getLogger(__name__)

def main():
    model = train()

    print("begin evaluation")

    Predict('data/test/test.nolabels.txt', model)


def train():

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

    SentenceList = SentenceList[int(len(SentenceList)/2):]

    ## Get Word Embeddings

    EmbeddingsList, LabelsList = CreateEmbeddings(SentenceList, LabelsList)

    print("finished Embedding")

    ## Create Model
    model = BiLSTM_CRF()

    print("model isntantiated")

    ## Move model to GPU if running with Cuda
    #model = model.cuda()

    ## Create Optimizer
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr= .5)

    ## Train for epochs using tqdm
    for i in tqdm(range(20), unit = 'epoch'):
    #for i in range(3):
        train_epoch(model, EmbeddingsList, LabelsList, optimizer)

    return model

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

    Period_Loss = 0

    for batch, labels, masks in tqdm(itertools.zip_longest(EmbeddingsList,LabelsList, MaskList), total = len(EmbeddingsList)):
    #for batch,labels,masks in itertools.zip_longest(EmbeddingsList,LabelsList, MaskList):
        emmissions = model(batch, masks, max_size)

        loss = -model.crf(emmissions,labels, mask = masks)

        Period_Loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.global_step += 1

    ## Save Epoch
    print("Loss: ", Period_Loss)



def Predict(filename, model):

    ## Read prediction data
    trainfile = open(filename,'r', encoding = 'utf8')

    SentenceListOrig = []

    currentSentence = []

    while True:
        line = trainfile.readline()

        if not line:
            SentenceListOrig.append(currentSentence)
            break

        if line != '\n':
            ## Add to current
            line = line.rstrip('\n')

            currentSentence.append(line)
        else:
            ## Add to list
            SentenceListOrig.append(currentSentence)

            currentSentence = []

    trainfile.close()

    print("file read")

    #Preprocess Data
    SentenceList= PreprocessSentencesT(SentenceListOrig.copy())

    print("pre processing done")

    ## Get Word Embeddings

    EmbeddingsList, wordDictionaries = CreateEmbeddingsT(SentenceList)

    print("word embedding done")

    model.eval()

    print("eval mode done")

    max_size = len(EmbeddingsList[0])

    MaskList = []
    for Embedding in EmbeddingsList:
        mask = []
        for word in Embedding:
            mask.append(0 if (sum(word) == 0) else 1)
        MaskList.append(mask)

    EmbeddingsList = torch.LongTensor(EmbeddingsList)
    MaskList = torch.ByteTensor(MaskList)

    TagList = []

    for sentence, masks in zip(EmbeddingsList, MaskList):
        emmissions = model(sentence.unsqueeze(0), masks.unsqueeze(0), max_size)

        tags_pred = model.crf.decode(emmissions,masks.unsqueeze(0))[0]

        TagList.append(tags_pred)

    tag2letter = {
        0 : 'O',
        1 : 'I',
        2 : 'B'
    }

    f = open(filename + "_Output", 'w')

    print("writting to file")

    for  sentence, tags, dict in tqdm(itertools.zip_longest(SentenceListOrig,TagList,wordDictionaries), total = len(SentenceListOrig)):
        for word in sentence:
            indexlist = []

            for key,value in dict.items():
                if word == value:
                    indexlist.append(key)

            if (len(indexlist) == 0):
                f.write('O')
                f.write('\n')
            else:
                consideredTags = []
                for index in indexlist:
                    consideredTags.append(tags[index])

                finaltag = tag2letter[max(consideredTags)]

                f.write(finaltag)
                f.write('\n')
        f.write('\n')
    f.close()


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()