import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from torchcrf import CRF

from allennlp.nn.util import replace_masked_values, masked_log_softmax, sort_batch_by_length

class BiLSTM_CRF(nn.Module):

    def __init__(self):
        super(BiLSTM_CRF,self).__init__()
        self.num_features = 1538
        self.num_classes = 3
        self.bilstm = nn.LSTM(self.num_features,64,batch_first = True, bidirectional = True)
        self.fc = nn.Linear(128,self.num_classes)
        self.crf = CRF(self.num_classes, batch_first = True)

        # Stores the number of gradient updates performed.
        self.global_step = 0

    def forward(self, batch, masks, max_size):

        batch = batch.type(torch.FloatTensor)

        # Get lengths per sequence

        lengths = []
        maskList = masks.tolist()
        for sentence in maskList:
            lengths.append(sum(sentence))

        # Sort by decreasing order of passage_lengths
        sortedEmbeddings, sortedsequence, restoration_indices, perm_passage = sort_batch_by_length(batch,torch.FloatTensor(lengths))

        # Pack padded sequence
        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence(sortedEmbeddings, sortedsequence, batch_first = True)

        # Feed into LSTM
        lstm_output, h_t = self.bilstm(pack_padded_sequence)

        # Pad packed sequence
        output_unpacked, unpacked_lens = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first = True, total_length = max_size)

        # Unsort unpacked sequence
        output_encoded = torch.index_select(output_unpacked,0,restoration_indices)

        # Feed into fc
        fc_output = self.fc(output_encoded)

        # Return emmissions
        return fc_output