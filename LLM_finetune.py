import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from dataset import AdultDataset,AdultDatasetGender
from peft import LoraModel,LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch.nn.functional as F
from torchcontrib.optim import SWA
import numpy as np
import random
import torch.nn as nn

def save(token_outputs,all_outputs,true_race,true_label,all_output_probs_1,all_output_probs_0,few_shot,fairness,fairprompt,filepath):
    #save the outputs
    np.save(f"{filepath}outputs_{few_shot}_{fairness}_{fairprompt}.npy",np.array(all_outputs))
    np.save(f"{filepath}tokens_{few_shot}_{fairness}_{fairprompt}.npy",np.array(all_outputs))
    np.save(f"{filepath}true_race_{few_shot}_{fairness}_{fairprompt}.npy",np.array(true_race))
    np.save(f"{filepath}true_label_{few_shot}_{fairness}_{fairprompt}.npy",np.array(true_label))
    np.save(f"{filepath}output_prob_1_{few_shot}_{fairness}_{fairprompt}.npy",np.array(all_output_probs_1))
    np.save(f"{filepath}output_prob_0_{few_shot}_{fairness}_{fairprompt}.npy",np.array(all_output_probs_0))

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

filepath = "mistral_output/"

set_seed(42)




def make_gumbel_softmax(emb):

    class GumbelSoftmax(torch.autograd.Function):
        E = emb

        @staticmethod
        def forward(ctx, input):
            L, N = input.shape
            with torch.enable_grad():
                softmaxed = F.gumbel_softmax(input, dim = 1)
            output  = torch.argmax(softmaxed, dim = 1)
            ctx.save_for_backward(input, softmaxed)
            return output, GumbelSoftmax.E(output)

        @staticmethod
        def backward(ctx, temp,grad_output):
            inp, softmaxed = ctx.saved_tensors
            grad_input = torch.autograd.grad(softmaxed, inp, grad_outputs=torch.mm(grad_output,GumbelSoftmax.E.weight.T))
            return grad_input
    
    return GumbelSoftmax


class CategoricalEmb(nn.Module):
    def __init__(self,emb):
        super(CategoricalEmb,self).__init__()
        fair_sent_dist = torch.randn(10,emb.weight.data.shape[0]).to("cuda")
        self.register_parameter("fair_sent_dist",nn.Parameter(fair_sent_dist))
        self.f_gumble_softmax = make_gumbel_softmax(emb)
        self.embeddings = emb
        self.embeddings.weight.requires_grad = False
        self.embeddings.to("cuda")

    def forward(self,input_ids,attn_mask):
        fairprompt_ids,fair_prompt = self.f_gumble_softmax.apply(self.fair_sent_dist)
        fair_prompt  = fair_prompt.repeat(input_ids.shape[0],1,1)
        fairprompt_ids = fairprompt_ids.repeat(input_ids.shape[0],1)
        embeddings = self.embeddings(input_ids)
        embeddings = torch.cat([fair_prompt,embeddings],dim=1)

        attention_mask = torch.cat([torch.ones_like(fairprompt_ids),attn_mask],dim=1)

        return embeddings,attention_mask



if __name__ == "__main__":


    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )


    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                            cache_dir="/research/hal-gaudisac/fairness/gpt_neo_cache",                                           
                                            bos_token="[BOS]",
                                            eos_token="[EOS]",
                                            unk_token="[UNK]",
                                            pad_token="[PAD]",
                                            mask_token="[MASK]")

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                                cache_dir="/research/hal-gaudisac/fairness/gpt_neo_cache",
                                                pad_token_id=tokenizer.eos_token_id,
                                                use_cache=False,
                                                )



    lora_model = get_peft_model(model, config)
    adv_model = CategoricalEmb(lora_model.get_input_embeddings())
    lora_model.to("cuda")
    adv_model.to("cuda")

    optimizer  = torch.optim.AdamW(lora_model.parameters(), lr=5e-4)
    adv_optimiser = SWA(torch.optim.Adam(adv_model.parameters(),lr=1e-1), swa_start=10, swa_freq=5, swa_lr=0.05)

    loss_fn = nn.CrossEntropyLoss()

    #Dataset
    few_shot =0
    fairness = 0.5
    fairprompt = False
    dataset = AdultDataset(few_shot=few_shot,number_of_samples=100,fairness=fairness,fairprompt=fairprompt)
    gender_dataset = AdultDatasetGender(few_shot=few_shot,number_of_samples=100,fairness=fairness,fairprompt=fairprompt)

    print(lora_model.print_trainable_parameters())
    print(sum(p.numel() for p in adv_model.parameters() if p.requires_grad))

    total_loss = 0
    accuracy = 0

    for epoch in range(10):
        lora_model.train()
        adv_model.eval()
        optimizer.zero_grad()
        for idx in range(len(gender_dataset)):
            prompts,race,label = gender_dataset[idx]
            #train mask language model
            model_inputs = tokenizer(prompts, return_tensors="pt",padding=True).to("cuda")
            output_ids = model_inputs.input_ids[:,-1]
            model_inputs.input_ids = model_inputs.input_ids[:,:-1]
            model_inputs.attention_mask = model_inputs.attention_mask[:,:-1]
            embeddings,attention_mask = adv_model(model_inputs.input_ids,model_inputs.attention_mask)
            logits = lora_model(inputs_embeds=embeddings,attention_mask=attention_mask).logits[:,-1,:].squeeze(1)
            loss = loss_fn(logits.view(-1, logits.size(-1)), output_ids.view(-1))
            loss.backward()
            with torch.no_grad():
                total_loss += loss.item()
                accuracy += torch.sum(torch.argmax(logits,dim=1) == output_ids).item() / len(output_ids)

            if (idx+1) % 9 == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        print("accuracy",accuracy/len(gender_dataset))
        print("total_loss",total_loss)
        
        
        accuracy = 0
        total_loss = 0
        
        adv_model.train() 
    
        lora_model.eval()
        for param in adv_model.parameters():
            param.grad = None
       
        for idx in range(len(dataset)):
            prompts,race,label = dataset[idx]
            #train mask language model
            model_inputs = tokenizer(prompts, return_tensors="pt",padding=True).to("cuda")
            output_ids = model_inputs.input_ids[:,-1]
            model_inputs.input_ids = model_inputs.input_ids[:,:-1]
            model_inputs.attention_mask = model_inputs.attention_mask[:,:-1]
            embeddings,attention_mask = adv_model(model_inputs.input_ids,model_inputs.attention_mask)
            logits = lora_model(inputs_embeds=embeddings,attention_mask=attention_mask).logits[:,-1,:].squeeze(1)
            loss =  -1* F.mse_loss(logits[0,:],logits[1,:])
            loss.backward()
            if (idx+1) % 9 == 0:
                adv_optimiser.step()
                #adv_optimiser is SWA and there are problems with SWA, therefore we need to manually update the grads to None
                for param in adv_model.parameters():
                    param.grad = None
                total_loss += loss.item()
        
        print("adv_total_loss",total_loss)
        total_loss = 0

        



### METRICS
with torch.no_grad():
    lora_model.eval()
    adv_model.eval()
    all_output_probs_1,all_output_probs_0= [],[]
    all_outputs,true_race,true_label = [],[],[]
    token_outputs = []
    for idx in range(len(dataset)):
        prompts,race,label = dataset[idx]
        #train mask language model
        model_inputs = tokenizer(prompts, return_tensors="pt",padding=True).to("cuda")
        output_ids = model_inputs.input_ids[:,-1]
        model_inputs.input_ids = model_inputs.input_ids[:,:-1]
        model_inputs.attention_mask = model_inputs.attention_mask[:,:-1]
        logits = lora_model(**model_inputs).logits[:,-1,:].squeeze(1)
        generated_ids = logits.argmax(dim=1)
        
        all_outputs.append(tokenizer.batch_decode(generated_ids,skip_special_tokens=True, clean_up_tokenization_spaces=False))
        output_prob_1 = torch.softmax(logits,dim=1)[:,657]
        output_prob_0 = torch.softmax(logits,dim=1)[:,352]
        all_output_probs_1.append(output_prob_1.cpu().numpy())
        all_output_probs_0.append(output_prob_0.cpu().numpy())
        true_race.append(race)
        true_label.append(label)

    
    save(token_outputs,all_outputs,true_race,true_label,all_output_probs_1,all_output_probs_0,few_shot,fairness,fairprompt,filepath)



with torch.no_grad():
    for i in range(10):
        fairprompt_ids,fair_prompt = adv_model.f_gumble_softmax.apply(adv_model.fair_sent_dist)
        print(" ".join(tokenizer.batch_decode(fairprompt_ids,skip_special_tokens=True, clean_up_tokenization_spaces=False)))



