import torch
from torch import nn
import torchvision

#############################
import os
import torch.optim
import torch.utils.data

from datasets import *
from utils import *
from torchtext.vocab import Vectors, GloVe
from transformers import *


##############################

from pytorch_pretrained_bert import BertTokenizer, BertModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.to(device)
model.eval()
# =============================================================================
data_folder = '/home/ajawalimalli/im_attention/outputs_imatt'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# =============================================================================

# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'rb') as j:
    word_map = json.load(j)

word_map_inverted = dict([[v,k] for k,v in word_map.items()])
############################################

############################################
text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer_bert.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

PAD = 0
START = 1
END = 2
UNK = 3
############################################


class Encoder(nn.Module):


    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):

        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out
    
    
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):

        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):


    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout,bert_dim,bert_att):

        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        
        #######################
        self.bert_att = bert_att
        
        if bert_att:
            self.embed_dim = bert_dim
        else:
            self.embed_dim = embed_dim
        ####################
        
        print("Decoder.........")
# =============================================================================
#         self.embed_dim = embed_dim
# =============================================================================
        
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        
        
# =============================================================================
#         self.init_weights()  # initialize some layers with the uniform distribution
#         self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
# 
#     def init_weights(self):
# 
#         self.embedding.weight.data.uniform_(-0.1, 0.1)
#         self.fc.bias.data.fill_(0)
#         self.fc.weight.data.uniform_(-0.1, 0.1)
# =============================================================================
        
        ##################################
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        if not bert_att:
            self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        ########################

    def load_pretrained_embeddings(self, embeddings):

        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):

        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):

        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
# =============================================================================
#         print("decoder forward")
# =============================================================================
 
    
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        ####
        max_dec_len = max(decode_lengths)
        ######
# =============================================================================
#         # Embedding
#         embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
# 
# =============================================================================
# =============================================================================
#         print("berting............")
# =============================================================================
        ####################################################
        if self.bert_att:
# =============================================================================
#             print("Entering bert")
# =============================================================================
# =============================================================================
#             print("Starting bert embeddings")
#             print()
# =============================================================================
            embeddings = []
            for cap_idx in  encoded_captions:
                
# =============================================================================
#                 print("No of encoded captions =", len(encoded_captions))
#                 print()
# =============================================================================
# =============================================================================
#                 print("padding")
# =============================================================================
                # padd caption to correct size

                '''
                ###############
                print("key_list")
                key_list = list(word_map.keys()) 
                val_list = list(word_map.values()) 
                #################    
                '''
# =============================================================================
#                 print("unfolding")
# =============================================================================

                    
                    
                cap = ' '.join([word_map_inverted[word_idx.item()] for word_idx in cap_idx])

            

# =============================================================================
#                 print("cap=",cap)
# =============================================================================

                
# =============================================================================
#                 print("Caption = ", cap)
# =============================================================================
# =============================================================================
#                 print("tokenizing")
# =============================================================================
                tokenized_cap = tokenizer_bert.tokenize(cap)  
                

                
                tokenized_cap = [x for x in tokenized_cap if x != 'pad']
                tokenized_cap = [x for x in tokenized_cap if x != 'start']
                tokenized_cap = [x for x in tokenized_cap if x != 'end']
                tokenized_cap = [x for x in tokenized_cap if x != '<']
                tokenized_cap = [x for x in tokenized_cap if x != '>']
                
                xx= '[CLS]'
                tokenized_cap = [xx] + tokenized_cap
                tokenized_cap.append('[SEP]')

# =============================================================================
#                 print("tokenized")                
# =============================================================================
# =============================================================================
#                 print('tokenized cap=', tokenized_cap)
# =============================================================================
                
                indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_cap)
                segments_ids =  [1] * len(indexed_tokens)
                #hugging face
# =============================================================================
#                 for tup in zip(tokenized_cap, indexed_tokens):
#                     print(tup)
# =============================================================================
                
# =============================================================================
#                 print(segments_ids)
# =============================================================================
                tokens_tensor = torch.tensor([indexed_tokens])#.to(device)
                segments_tensors = torch.tensor([segments_ids])#.to(device)
# =============================================================================
#                 
#                 print("tokens_tensor = " ,tokens_tensor
#                       )
# =============================================================================

                
                with torch.no_grad():
                    encoded_layers, _ = model(tokens_tensor.to(device),segments_tensors.to(device))
    
# =============================================================================
#                 bert_embedding = encoded_layers[11].squeeze(0)
# =============================================================================
# =============================================================================
#                 encoded_layers = torch.FloatTensor(encoded_layers)   
# =============================================================================
# =============================================================================
#                 print(encoded_layers)    
#                 print(encoded_layers[11].size())
# =============================================================================
# =============================================================================
#                 print("berted")
# =============================================================================
                '''
                
                split_cap = cap.split()
# =============================================================================
#                 print("split_cap = ", split_cap)
# =============================================================================
                tokens_embedding = []
                j = 0
    
                for full_token in split_cap:
                    curr_token = ''
                    x = 0
                    for i,_ in enumerate(tokenized_cap[1:-1]): # disregard CLS
                        token = tokenized_cap[i+j]
                        piece_embedding = bert_embedding[i+j]
                        
                        # full token
# =============================================================================
#                         print("token =",token)
#                         print("full_token= ",full_token)
# =============================================================================
                        if token == full_token and curr_token == '' :
                            tokens_embedding.append(piece_embedding)
                            j += 1
                            break
                        else: # partial token
                            x += 1
                            
                            if curr_token == '':
                                tokens_embedding.append(piece_embedding)
                                curr_token += token.replace('#', '')
                            else:
                                tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                curr_token += token.replace('#', '')
                                
                                if curr_token == full_token: # end of partial
                                    j += x
                                    break                            
    
                cap_embedding = torch.stack(tokens_embedding)
                
                print("iiii= ",cap_embedding.shape)
                embeddings.append(cap_embedding)
                '''
                
# =============================================================================
#                 embeddings.append(bert_embedding)
# =============================================================================
                
# =============================================================================
#                 print("length of embeddings = ", len(embeddings))
# =============================================================================
                
# =============================================================================
#             embeddings = torch.stack(embeddings)
# =============================================================================
            embeddings = encoded_layers[11]
        
        else:
            embeddings = self.embedding(encoded_captions)  
        ####################################################
# =============================================================================
#         print("Got embeddings")
# =============================================================================
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)



        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
# =============================================================================
#         print("am here now")
# =============================================================================
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
# =============================================================================
#         print("For loop done")    
# =============================================================================

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

