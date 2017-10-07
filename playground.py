import torch
from torch import nn
from torch.autograd import Variable
from dataset import GRASS
from model import GRASSEncoder
from model import GRASSDecoder
import model
import torch.utils.data
import torchfold
from time import time
import util
import matplotlib.pyplot as plt
from draw3dOBB import showGenshape

def class_collate(batch):
    return batch

config = util.get_args()
m = GRASSEncoder(config)
m2 = GRASSDecoder(config)
m.cuda()
m2.cuda()

encoder = torch.load('encoder.pkl')
decoder = torch.load('decoder.pkl')

#test = Variable(torch.rand(1,80)).cuda()
#boxes = model.decode_structure(decoder, test)
#showGenshape(torch.cat(boxes,0).data.cpu().numpy())

grassData = GRASS('data')
dataloader = torch.utils.data.DataLoader(grassData, batch_size=123, shuffle=False, collate_fn=class_collate)

for i, batch in enumerate(dataloader):
    fold = torchfold.Fold(cuda=True)
    res = []
    for example in batch:
        res.append(model.encode_structure_fold(fold, example))
    res = fold.apply(encoder, [res])
    res1 = torch.split(res[0], 1, 0)
    for example, f in zip(batch, res1):
        ff, kld = torch.chunk(f, 2, 1)
        boxes = model.decode_structure(decoder, ff)
        showGenshape(torch.cat(boxes,0).data.cpu().numpy())
   

optimizer1 = torch.optim.Adam(m.parameters(), lr=1e-3)
optimizer2 = torch.optim.Adam(m2.parameters(), lr=1e-3)

errs = []
for epcho in range(250):
    for i, batch in enumerate(dataloader):
        fold = torchfold.Fold(cuda=True)
        res = []
        for example in batch:
            res.append(model.encode_structure_fold(fold, example))
        res = fold.apply(m, [res])
        res1 = torch.split(res[0], 1, 0)
        ttt = []
        fold = torchfold.Fold(cuda=False)
        kldLoss = []
        for example, f in zip(batch, res1):
            ff, kld = torch.chunk(f, 2, 1)
            ttt.append(model.decode_structure_fold(fold, ff, example))
            kldLoss.append(kld)
        ttt = fold.apply(m2, [ttt, kldLoss])
        err = ttt[0].sum() + ttt[1].sum().mul(-0.05)
        err = err/len(batch)
        m.zero_grad()
        m2.zero_grad()
        err.backward()
        optimizer1.step()
        optimizer2.step()

        errs.append(err.data[0])
        if i % 5 == 0 :
            print(err.data)
            plt.plot(errs, c='#4AD631')
            plt.draw()
            plt.pause(0.01)

torch.save(m, 'encoder.pkl')
torch.save(m2, 'decoder.pkl')
        
