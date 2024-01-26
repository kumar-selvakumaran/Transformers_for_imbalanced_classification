
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
    """
    The function calculates the accuracy of the predictions by comparing them to the actual labels.
    
    :param preds: The `preds` parameter is a numpy array containing the predicted values. It should have
    a shape of (batch_size, num_classes), where `batch_size` is the number of samples and `num_classes`
    is the number of classes or categories
    :param labels: The "labels" parameter is a numpy array that contains the true labels for a set of
    examples. Each element in the array represents the true label for a single example
    :return: the flat accuracy, which is the ratio of correct predictions to the total number of
    predictions.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    """
    The function takes a time in seconds and returns a formatted string in the format hh:mm:ss.
    
    :param elapsed: The parameter "elapsed" represents the time in seconds that you want to format
    :return: a string in the format "hh:mm:ss" representing the elapsed time.
    """
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def convert_labels(og_labels):
    """
    The function `convert_labels` takes in a list of original labels, converts them to unique and sorted
    labels, assigns unique IDs to each label, and returns a new list of labels where each label is
    replaced with its corresponding ID.
    
    :param og_labels: The `og_labels` parameter is a list or array containing the original labels that
    you want to convert
    :return: a list of new labels, where each label in the original labels list is replaced with its
    corresponding ID.
    """
    labels = np.unique(np.array(og_labels))
    labels.sort()
    ids = range(0,len(labels))
    label_to_id_dict = dict(zip(labels, ids))
    new_labels = [*map(label_to_id_dict.get, og_labels.tolist())]
    return new_labels

    """
    The function "getlbl2id" takes in a list of original labels, sorts them, assigns unique IDs to each
    label, and returns a dictionary mapping each label to its corresponding ID.
    
    :param og_labels: The parameter "og_labels" is expected to be a list or array containing the
    original labels. These labels can be of any data type, such as strings or integers
    :return: a dictionary that maps each unique label in the input `og_labels` to a unique ID.
    """
def getlbl2id(og_labels):
    labels = np.unique(np.array(og_labels))
    labels.sort()
    ids = range(0,len(labels))
    label_to_id_dict = dict(zip(labels, ids))
    return label_to_id_dict
                           

# The `trainer` class is used to prepare data and train a model for sequence classification using the
# Deberta architecture.

class trainer:
    def __init__(self):
        """
        The function initializes the device to be used for computation, either CUDA if available or CPU
        otherwise.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    """
    The `prepare_data` function takes in the filenames of tokenized train and validation sets, batch
    size, and prepares the data for training and validation by loading the tokenized data, converting
    labels, creating tensors, and creating data loaders.
    
    :param train_set_tokens_filename: The `train_set_tokens_filename` parameter is the file name of the
    pickle file that contains the tokenized training data. This file is typically generated in a
    previous step of the data preparation process, such as tokenization
    :type train_set_tokens_filename: str
    :param valid_set_tokens_filename: The `valid_set_tokens_filename` parameter is the filename of a
    pickle file that contains the tokenized validation set data. This file should have been created
    using the `tokenization.ipynb` notebook
    :type valid_set_tokens_filename: str
    :param batch_size: The batch_size parameter determines the number of samples that will be processed
    in each iteration during training. It specifies how many samples will be loaded into memory at once,
    defaults to 128
    :type batch_size: int (optional)
    """
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
        
        """
        The `train` function trains a model using the specified optimizer, learning rate, and other
        parameters, and saves the training results and model checkpoints.
        
        :param model_name: The name of the model to be used for training. It should be a string that
        represents the model architecture, such as "microsoft/deberta-base"
        :param optimizer_name: The `optimizer_name` parameter is the name of the optimizer that will be
        used for training the model. It should be an initialized optimizer object from the
        `transformers.optimization` module
        :param learning_rate: The learning rate is a hyperparameter that determines the step size at each
        iteration while updating the model parameters during training. It controls how quickly or slowly
        the model learns from the training data
        :param epsilon: The `epsilon` parameter is used in the AdamW optimizer. It is a small value added
        to the denominator to improve numerical stability when dividing by the square root of the second
        moment of the gradients. It prevents division by zero
        :param number_of_epochs: The parameter `number_of_epochs` specifies the number of times the
        training loop will iterate over the entire dataset. Each iteration is called an epoch. By
        default, it is set to 2, defaults to 2
        :type number_of_epochs: int (optional)
        :param warmup: The `warmup` parameter is used to specify the number of warmup steps during
        training. Warmup steps are a period at the beginning of training where the learning rate is
        gradually increased from a very small value to the desired learning rate. This helps the model to
        stabilize and avoid large changes in the, defaults to 3
        :type warmup: int (optional)
        :param use_focal_loss: A boolean flag indicating whether to use focal loss or not. Focal loss is
        a modification of the standard cross-entropy loss that focuses on hard examples during training.
        It can be useful in imbalanced classification tasks where the majority class dominates the loss
        function, defaults to False
        :type use_focal_loss: bool (optional)
        """
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
            