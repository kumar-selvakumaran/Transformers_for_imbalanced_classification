
import pandas as pd
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch
import datetime
import pickle
import time

from transformers import AdamW,DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import cross_entropy

from .data_utils import set_seed, make_path


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def convert_labels(og_labels):
    labels = np.unique(np.array(og_labels))
    labels.sort()
    ids = range(0,len(labels))
    label_to_id_dict = dict(zip(labels, ids))
    new_labels = [*map(label_to_id_dict.get, og_labels.tolist())]
    return new_labels

def getlbl2id(og_labels):
    labels = np.unique(np.array(og_labels))
    labels.sort()
    ids = range(0,len(labels))
    label_to_id_dict = dict(zip(labels, ids))
    return label_to_id_dict
                           


class trainer:
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def prepare_data(self,
                     train_set_tokens_filename: str, #pkl file of tokens as made in tokenization.ipynb
                     valid_set_tokens_filename: str, # pkl ...
                     batch_size: int = 128,
                     ):
        
        self.model_name = train_set_tokens_filename.split("_")[-3]
        
        
        with open(train_set_tokens_filename, 'rb') as f:
            train_encodings = pickle.load(f)

        with open(valid_set_tokens_filename, 'rb') as f:
            validation_encodings= pickle.load(f)

        validation_labels = validation_encodings["labels"]    
        id2lbl = getlbl2id(validation_labels)
            
        train_inputs = train_encodings["input_ids"]  
        train_masks = train_encodings["attention_masks"]

        train_labels = train_encodings["labels"]
        train_labels = convert_labels(train_labels)

        validation_inputs = validation_encodings["input_ids"]
        validation_masks = validation_encodings["attention_masks"]

        validation_labels = validation_encodings["labels"]
        validation_labels = convert_labels(validation_labels)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        validation_labels = torch.tensor(validation_labels, dtype=torch.long)
        train_masks = torch.tensor(train_masks, dtype=torch.long)
        validation_masks = torch.tensor(validation_masks, dtype=torch.long)

        num_labels = len(np.unique(np.array(train_labels)))

        self.num_labels = num_labels

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)  

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def train(self,
              model_name,
              optimizer_name, #an initialized optimizer from transformer.optimization
              learning_rate,
              epsilon,
              number_of_epochs: int = 2,
              warmup: int = 3,
              use_focal_loss: bool = False):
        

        #<<<<<<<<<<<<<   create model dict for this should use model_name >>>>>>>>>>>>>
        self.model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels = self.num_labels)

        if self.device.type == "cuda":
            self.model.cuda()

        #<<<<<<<<<<<<<   create optimizer dict for this should use optimizer_name >>>>>>>>>>>>>
        self.optimizer = AdamW(self.model.parameters(),
                  lr = learning_rate, #args.learning_rate
                  eps = epsilon  #args.adam_epsilon
                )
        
        self.num_epochs = number_of_epochs
        self.warmup = warmup
        total_steps = len(self.train_dataloader) * self.num_epochs
        self.scheduler = scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                            num_warmup_steps = self.warmup, 
                                            num_training_steps = total_steps)


        if self.model_name not in str(type(self.model)):
            print(f"##############\ndata and model may not match, model name acc to data is {self.model_name}, but model intialized may not be {self.model_name}\n###########")


        
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2

        set_seed()

        loss_values = []
        loss_values = []
        gran_train_loss_values = []
        gran_val_loss_values = []
        gran_train_accuracy_values = []
        gran_val_accuracy_values = []

        # For each epoch...
        for epoch_i in range(0, self.num_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.num_epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            gran_loss = 0
            train_accuracy = 0
            train_preds=[]
            train_true=[]
            nb_train_steps, nb_train_examples = 0, 0

            self.model.train()

            for step, batch in enumerate(self.train_dataloader):

                # Progress update every 100 batches.
                if step % 50 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()        

                outputs = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
                
                loss = outputs[0]
                
                # Accumulate the training loss over all of the batches 
                ###########################################
                #
                if self.use_focal_loss:
                    ce_loss  = cross_entropy(outputs[1].view(-1, self.model.num_labels), b_labels, reduction = 'none')
                    pt = torch.exp(-ce_loss) # ' - ' (minus) is important
                    focal_loss = (self.focal_loss_alpha * (1-pt)**self.focal_loss_gamma * ce_loss).mean()
                    loss = focal_loss
                    #print(f"outputs loss : {(type(outputs[0]), outputs[0].shape), loss}, extracted loss : {(type(focal_loss), ce_loss.shape)}")

                ############################################
                batch_loss = loss.item()
                total_loss += batch_loss
                gran_loss += batch_loss
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                scheduler.step()
                
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                tmp_train_accuracy = flat_accuracy(logits, label_ids)
                train_accuracy += tmp_train_accuracy
                nb_train_steps += 1
                
                if step % 300 == 0 and not step == 0: ################### GRANULAR VALIDATION ###################### 
                    gran_train_loss_values.append(gran_loss / nb_train_steps)
                    gran_train_accuracy_values.append(train_accuracy/nb_train_steps)
                    
                    self.model.eval()
                    
                    preds=[]
                    true=[]
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    for vbatch in self.validation_dataloader:
                        vbatch = tuple(t.to(self.device) for t in vbatch)
                        b_input_ids, b_input_mask, b_labels = vbatch
                        with torch.no_grad():        

                            outputs = self.model(b_input_ids, 
                                            token_type_ids=None, 
                                            attention_mask=b_input_mask,
                                            labels=b_labels)
                            
                        v_loss = outputs[0]
                        if self.use_focal_loss:
                            v_ce_loss  = cross_entropy(outputs[1].view(-1, num_labels), b_labels, reduction = 'none')
                            v_pt = torch.exp(-v_ce_loss) # ' - ' (minus) is important
                            v_focal_loss = (self.focal_loss_alpha * (1- v_pt)**self.focal_loss_gamma * v_ce_loss).mean()
                            v_loss = v_focal_loss
                            
                        eval_loss += v_loss  
                        
                        logits = outputs[1]
                        logits = logits.detach().cpu().numpy()
                        label_ids = b_labels.to('cpu').numpy()

                        #preds.append(logits)
                        #true.append(label_ids)
                        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                        eval_accuracy += tmp_eval_accuracy
                        nb_eval_steps += 1

                    eval_loss = eval_loss / len(self.validation_dataloader) 
                    gran_val_loss_values.append(eval_loss)
                    gran_val_accuracy_values.append(eval_accuracy/nb_eval_steps)
                    print("\n###  training accuracy: {0:.2f}  ###".format(train_accuracy/nb_train_steps))
                    print("###  validation accuracy: {0:.2f}  ###".format(eval_accuracy/nb_eval_steps))
                    print(f"### current training loss : {gran_loss / nb_train_steps}  ###")
                    print(f"### current validation loss : {eval_loss}  ###\n")
                    
                    gran_loss = 0
                    nb_train_steps, train_accuracy = 0, 0

            avg_train_loss = total_loss / len(self.train_dataloader)            
            
            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
            
            
        print("")
        print("Training complete!")        

        make_path("./training_results/")

        make_path("./training_results/{self.model_name}")

        protov3_1epoc_results = {
            "train accuracy" : gran_train_accuracy_values,
            "validation accuracy" : gran_val_accuracy_values,
            "train loss" : gran_train_loss_values,
            "validation loss" : gran_val_loss_values
        }

        self.model.save_pretrained("./training_results/{self.model_name}/{self.model_name}_model", from_pt=True) 

        with open(f"./training_results/{self.model_name}/protov2_{self.model_name}_{'FOCAL_' if self.use_focal_loss else ''}fixed_{self.num_epochs}epoc_results.pkl", 'wb') as f:
            pickle.dump(protov3_1epoc_results, f)
            