import torch
from torch import nn
from torch.autograd import Variable
from dataset import GRASS
from model import GRASSEncoder
from model import GRASSDecoder
import model
import torch.utils.data
import torchfold
import util

def class_collate(batch):
    return batch

config = util.get_args()
encoder = GRASSEncoder(config)
decoder = GRASSDecoder(config)
encoder.cuda()
decoder.cuda()

grassData = GRASS('data')
dataloader = torch.utils.data.DataLoader(grassData, batch_size=123, shuffle=True, collate_fn=class_collate)

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=1e-3)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=1e-3)

for epcho in range(500):
    if epcho % 100 == 0 and epcho != 0 :
        torch.save(encoder, 'VAEencoder.pkl')
        torch.save(decoder, 'VAEdecoder.pkl')
    for i, batch in enumerate(dataloader):
        fold = torchfold.Fold(cuda=True, variable=False)
        encoding = []
        for example in batch:
            encoding.append(model.encode_structure_fold(fold, example))
        encoding = fold.apply(encoder, [encoding])
        encoding = torch.split(encoding[0], 1, 0)
        decodingLoss = []
        fold = torchfold.Fold(cuda=True, variable=True)
        kldLoss = []
        for example, f in zip(batch, encoding):
            ff, kld = torch.chunk(f, 2, 1)
            decodingLoss.append(model.decode_structure_fold(fold, ff, example))
            kldLoss.append(kld)
        decodingLoss = fold.apply(decoder, [decodingLoss, kldLoss])
        err_re = decodingLoss[0].sum()/len(batch)
        err_kld = decodingLoss[1].sum().mul(-0.05)/len(batch)
        err = err_re + err_kld
        encoder.zero_grad()
        decoder.zero_grad()
        err.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        if i % 5 == 0 :
            print("reconstruction_error: %s; KLD_error: %s" % (str(err_re.data[0]), str(err_kld.data[0])))

torch.save(encoder, 'VAEencoder.pkl')
torch.save(decoder, 'VAEdecoder.pkl')