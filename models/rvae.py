import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

params = {
        'BATCH_SIZE'   : 8,
        'MAX_SEQ_LEN'  : 20,
        'vocab_size'   : None,
        'EMB_DIM'      : 100,
        'ENC_HDIM'     : 100,
        'DEC_HDIM'     : 100,
        'ZDIM'         : 20,
        'DROP_PROB'    : 0.4
        }
        
class RVAE(nn.Module):
    def __init__(self, params):
        super(RVAE, self).__init__()

        self.params = params
        # word embedding
        self.wemb = nn.Embedding(self.params['vocab_size'], self.params['EMB_DIM'])
        # encoder
        self.encoder_rnn = nn.LSTM(
                input_size=self.params['EMB_DIM'],
                hidden_size=self.params['ENC_HDIM'],
                num_layers=1,
                batch_first=True
                )
        # variational parameters
        self.context_to_mu = nn.Linear(2*params['ENC_HDIM'], params['ZDIM'])
        self.context_to_logsigma = nn.Linear(2*params['ENC_HDIM'], params['ZDIM'])
        # decoder
        self.decoder_rnn = nn.LSTM( # word embedding dim + dim(z)
                input_size=self.params['EMB_DIM'] + self.params['ZDIM'],
                hidden_size=self.params['DEC_HDIM'],
                num_layers=1,
                batch_first=True
                )
        # project to vocab space
        self.fc_o = nn.Linear(self.params['DEC_HDIM'], self.params['vocab_size'])


    def encode(self, x):
        [batch_size, seqlen, emb_size] = x.size()
        # encode with RNN
        _, final_state = self.encoder_rnn(x)
        return torch.cat(final_state, dim=-1).view(-1, self.params['ENC_HDIM']*2)

    def reparameterize(self, mu, logsigma):
        if self.training:
            std = torch.exp(0.5*logsigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x, z, initial_state=None):
        [batch_size, seqlen, _ ] = x.size()
        x = F.dropout(x, self.params['DROP_PROB'])
        # tile z : [ batch_size x seq_len x zdim ]
        z = torch.cat([z] * seqlen, dim=1).view(batch_size, seqlen, self.params['ZDIM'])
        # decoder input
        x = torch.cat([x,z], dim=-1)
        # decoder RNN
        decoder_out, decoder_state = self.decoder_rnn(x, initial_state)
        decoder_out =  decoder_out.contiguous().view(-1, self.params['DEC_HDIM'])
        # project to vocab space
        return self.fc_o(decoder_out).view(
                batch_size, seqlen, self.params['vocab_size'])

    def forward(self, enc_input, dec_input):
        batch_size, _ = enc_input.size()
        context = self.encode(self.wemb(enc_input))
        z = self.reparameterize(
                self.context_to_mu(context),
                self.context_to_logsigma(context)
                )
        #z = Variable(torch.randn([batch_size, self.params['ZDIM']])).cuda()
        #z = torch.randn([batch_size, self.params['ZDIM']])
        return self.decode(self.wemb(dec_input), z)


if __name__ == '__main__':
    params['vocab_size'] = 150
    model = RVAE(params)

    # random integer tensor
    ip_seq = torch.LongTensor(params['BATCH_SIZE'], 12)
    ip_seq.random_(0, params['vocab_size'])

    op_seq = torch.LongTensor(params['BATCH_SIZE'], 12)
    op_seq.random_(0, params['vocab_size'])

    out = model(ip_seq, op_seq)
    print(out.size())
