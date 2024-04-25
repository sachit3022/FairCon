import torch
from torch.nn import functional as F
from torchcontrib.optim import SWA

from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class GumbelSoftmax(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        L, N = input.shape
        with torch.enable_grad():
            softmaxed = F.gumbel_softmax(input, dim = 1)
        output    = torch.argmax(softmaxed, dim = 1)
        ctx.save_for_backward(input, softmaxed)
        return output, E(output)

    @staticmethod
    def backward(ctx, temp,grad_output):
        input, softmaxed = ctx.saved_tensors
        grad_input = torch.autograd.grad(softmaxed, input, grad_outputs=torch.mm(grad_output,E.weight.T))
        return grad_input

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E = nn.Embedding(500,2048,device=DEVICE)
E.requires_grad=False

if __name__ == "__main__":
    #given a distribution of tokens ask gumbel to generate the same token

    ##test the training of gumbel softmax
    fair_sent_dist = torch.randn(1,500,device=DEVICE)
    #fair_sent_dist[0,1] = 1


    
    fair_sent_dist.requires_grad = True
    base_opt = torch.optim.AdamW([fair_sent_dist],lr=1)
    opt = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
   
    ground_truth = torch.clone(E(torch.tensor([0],device=DEVICE)).detach())

    ground_truth.requires_grad = False
    losses = []
    for i in tqdm(range(1000)):
        emb = F.gumbel_softmax(fair_sent_dist,dim=1,tau=1,hard=False)
        loss = F.mse_loss(emb@ E.weight.data ,ground_truth)
        fair_sent_dist.grad = None
        loss.backward()
        #get the max abs of gradient
        #with torch.no_grad():
        #    print(fair_sent_dist.grad.abs().max())

        opt.step()
        #scheduler.step()
        losses.append(loss.item())
    plt.plot(losses)
    plt.savefig("gumbel_loss.png")
    
    print(F.softmax(fair_sent_dist,dim=1))
    

