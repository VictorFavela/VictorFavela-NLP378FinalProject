import emoji
import re

def PreprocessSentences(sentences, labels):
    for i in range(0,len(sentences)):

        ## Remove RT
        if (sentences[i][0] == 'RT'):
            sentences[i] = sentences[i][3:]

        ## Regex for removing url's
        regex = re.compile(r'(http)')

        ## Remove url's and Emojis
        sentences[i], labels[i] = zip(*((word,label) for word,label in zip(sentences[i],labels[i]) 
                                        if ((not regex.search(word)) and (not emoji.get_emoji_regexp().search(word))) ))

    return sentences, labels

def PreprocessSentencesT(sentences):
    for i in range(0,len(sentences)):

        ## Remove RT
        if (sentences[i][0] == 'RT'):
            sentences[i] = sentences[i][3:]

        ## Regex for removing url's
        regex = re.compile(r'(http)')

        ## Remove url's and Emojis
        sentences[i] = [word for word in sentences[i] if ((not regex.search(word)) and (not emoji.get_emoji_regexp().search(word)))]

    return sentences