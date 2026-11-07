# GenAI acknowledgement: I acknowledge the use of Generative AI
# to review drafts of my work and provide feedback. All code was written by me.

import json
import csv
from dataclasses import dataclass, asdict, fields

FIELD_TYPES = {"model_name":str,"training_accuracy":float,"training_loss":float,"validation_accuracy":float,
               "validation_loss":float,"epochs":int,"batch_size":int,"learning_rate":float}

def merge(left,right,key=lambda x: x,reverse=False):
    result = []
    i = 0
    j = 0
    while i < len(left)  and j < len(right):
        if reverse:
            if key(left[i]) >= key(right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        else:
            if key(left[i]) <= key(right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def merge_sort(items,key=lambda x: x,reverse=False):
    if items == [] or len(items) <= 1:
        return items
    mid = len(items)//2
    left = items[:mid]
    right = items[mid:]
    left_half = merge_sort(left,key,reverse)
    right_half = merge_sort(right,key,reverse)
    result = merge(left_half,right_half,key,reverse)
    return result

@dataclass
class Model:
    model_name: str
    training_accuracy: float
    training_loss: float
    validation_accuracy: float
    validation_loss: float
    epochs: int
    batch_size: int
    learning_rate: float

    def __post_init__(self):
        #Model name check
        if not isinstance(self.model_name,str) or not self.model_name.strip():
            raise ValueError("Enter a valid model name")
        
        #Training accuracy check
        if not isinstance(self.training_accuracy,(float,int)) or not 0 <= self.training_accuracy <= 100:
            raise ValueError("Enter a valid training accuracy in a range of 0 - 100")
        
        #Training loss check
        if not isinstance(self.training_loss, (float,int)) or not self.training_loss >= 0:
            raise ValueError("Training loss cannot be negative. Please enter a valid training loss")
        
        #Validation Accuracy check
        if not isinstance(self.validation_accuracy,(float,int)) or not 0 <= self.validation_accuracy <= 100:
            raise ValueError("Enter a valid validation accuracy in a range of 0 - 100")
        
        #Validation loss check
        if not isinstance(self.validation_loss, (float,int)) or not self.validation_loss >= 0:
            raise ValueError("Validation loss cannot be negative. Please enter a valid validation loss")
        
        #No of epochs check
        if not isinstance(self.epochs,int) or  self.epochs <= 0 : 
            raise ValueError("Please enter valid epochs numbers")
        
        #No of batch size check
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Please enter a valid batch size")
        
        #Learning rate check
        if not isinstance(self.learning_rate, float) or  self.learning_rate <= 0: 
            raise ValueError("Please enter a valid learning rate")

class ModelManager:
    def __init__(self):
        self.models = {}
        self.next_id = 1

    def add_model(self,model):
        if isinstance(model,Model):
            run_id = self.next_id
            self.models[run_id] = model
            self.next_id += 1
            return run_id
        else:
            raise TypeError("Please enter a valid model run")

    def delete_model(self,run_id):
        if run_id not in self.models.keys():
            raise KeyError(f"A model with ID {run_id} doesn't exist")
        del self.models[run_id]

    def update_model(self,run_id,new_model):
        if not isinstance(new_model,Model):
            raise TypeError("Please enter a valid model run")
        if run_id not in self.models.keys():
            raise KeyError(f"A model with ID {run_id} doesn't exist")
        self.models[run_id] = new_model

    def get_models(self):
        return self.models.copy()

    def save_json(self,path):
        runs = [asdict(model) for model in self.models.values()]
        with open(path,"w",encoding="utf-8") as f:
            json.dump(runs,f,indent=4)

    def load_json(self,path):
        counter = 1
        temp_models = {}
        with open(path,"r",encoding="utf-8") as f:
            runs = json.load(f)
        for run in runs:
            temp_models[counter] = (Model(**run))
            counter += 1
        self.models = temp_models
        self.next_id = counter

    def save_csv(self,path):
        names = [field.name for field in fields(Model)]
        with open(path,"w",encoding="utf-8",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=names)
            writer.writeheader()
            for model in self.models.values():
                writer.writerow(asdict(model))

    def load_csv(self,path):
        new_models = {}
        counter = 1
        with open(path,"r",newline="",encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted_run = {key:FIELD_TYPES[key](value) for key,value in row.items()}
                new_models[counter] = Model(**converted_run)
                counter += 1 
        self.models = new_models
        self.next_id = counter

    def sort_runs(self,metric):
        if metric not in ("validation_accuracy","validation_loss"):
            raise ValueError("Please enter either Validation Accuracy or Validation Loss")
        reversed = True if metric == "validation_accuracy" else False
        runs = list(self.get_models().items())
        sorted_runs = merge_sort(items=runs,key=lambda x: getattr(x[1],metric),reverse=reversed)
        return sorted_runs