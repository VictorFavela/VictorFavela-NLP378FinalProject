from itertools import repeat

from names_dataset import NameDataset
import nltk
nltk.download('averaged_perceptron_tagger')

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

def CreateGazetter(sentences):
    gazetterList = []

    m = NameDataset()

    for sentence in sentences:
        currentSentence = []

        for word in sentence:
            currentSentence.append(((m.search_first_name(word) + m.search_last_name(word)) / 200))
        
        gazetterList.append(currentSentence)
    
    return gazetterList

def CreatePOSTagging(sentences):
    posList = []

    for sentence in sentences:
        currentSentence = []
        for word in sentence:
            currentSentence.append((1 if nltk.pos_tag(word)[0][1] == 'NNP' else 0) )
        posList.append(currentSentence)

    return posList

def CreateEmbeddings(sentences, labels):
    ## Create Embeddings list and corresponding labels:
    ## Embeddings List - Matrix of shape (number of sentences, 64, 1 + 1 + BertSize * 2)
    ## Labels List - Matrix of shape (number of sentences, 64)

    ## Create Gazetter Encoding
    gazetterList = CreateGazetter(sentences)

    ## Create POS-Tagging Encoding
    posList = CreatePOSTagging(sentences)

    ## Generate distilBert word Tokenizations
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    distilBertModel = DistilBertModel.from_pretrained('distilbert-base-cased', output_hidden_states = True)
    distilBertModel.eval()

    wordEmbeddings = []
    wordLabels = []

    for sentence,label,gazetter,pos in zip(sentences,labels,gazetterList,posList):
        sentenceL = []
        labelL = []
        gazetterL = []
        posL = []

        for word,lab,gazet,p in zip(sentence,label,gazetter,pos):

            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            word_id = tokenizer.convert_tokens_to_ids(tokenized_word)
            sentenceL.extend(word_id)

            labelL.extend(repeat(lab , n_subwords))
            gazetterL.extend(repeat(gazet , n_subwords))
            posL.extend(repeat(p , n_subwords))

        wordLabels.append(labelL)

        sentenceL = torch.LongTensor(sentenceL)
        gazetterL = torch.LongTensor(gazetterL).unsqueeze(1)
        posL = torch.LongTensor(posL).unsqueeze(1)
        
        #Feed into Distilbert

        output = distilBertModel(sentenceL.unsqueeze(0))

        last_hidden = output.last_hidden_state.squeeze()

        second_hidden = output.hidden_states[5].squeeze()

        wordEmbedding = torch.cat((gazetterL,posL,last_hidden,second_hidden),1)

        wordEmbeddings.append(wordEmbedding)

    wordEmbeddings_padded = pad_sequence(wordEmbeddings, batch_first = True)

    return wordEmbeddings_padded, wordLabels