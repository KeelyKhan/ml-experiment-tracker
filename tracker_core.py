# GenAI acknowledgement: I acknowledge the use of Claude AI (Anthropic, https://claude.ai)
# to review drafts of my work and provide feedback. All code was written by me.

import json
import csv
from dataclasses import dataclass, asdict, fields

FIELD_TYPES = {"model_name":str,"training_accuracy":float,"training_loss":float,"validation_accuracy":float,
               "validation_loss":float,"epochs":int,"batch_size":int,"learning_rate":float}

SORTABLE_METRICS = ("validation_accuracy", "validation_loss")

MILD_GAP = 3
SERIOUS_GAP = 5
LOW_ACCURACY = 75
HIGH_LEARNING_RATE = 0.1
LOW_LEARNING_RATE = 0.001
MINIMUM_EPOCHS = 5

OVERFITTING_SUGGESTION = """
 - Implement regularization techniques such as implementing dropout layer or weight decay
 - Data augmentation
 - Early stopping of model training
 - Trying  a simpler model
"""
UNDERFITTING_SUGGESTION = """
- Train the model for more epochs
- Use a more advanced model
- Reduce the regularisation
- Increase the learning rate
"""

HIGH_LR_SUGGESTION = """
- Lower the learning rate for stable learning
"""

LOW_LR_SUGGESTION = """
- Increase the learning rate
- Train for more epochs
"""

LOW_EPOCHS_SUGGESTION = """
- Train for more epochs
"""

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

def mean(values):
    if not values:
        raise ValueError("Mean cannot be computed on empty list")
    return sum(values)/len(values)

def variance(values):
    avg = mean(values)
    squared_diffs = [(value - avg) ** 2 for value in values]
    return mean(squared_diffs)

def std_dev(values):
    return (variance(values))**0.5

def get_recommendations(run):
    recommendations = []
    gap = run.training_accuracy - run.validation_accuracy
    if gap > SERIOUS_GAP:
        recommendations.append({"issue": "Overfitting (Serious)", "suggestion": OVERFITTING_SUGGESTION})
    elif gap > MILD_GAP:
        recommendations.append({"issue": "Overfitting (Mild)", "suggestion": OVERFITTING_SUGGESTION})

    if run.training_accuracy < LOW_ACCURACY and run.validation_accuracy < LOW_ACCURACY:
        recommendations.append({"issue": "Underfitting", "suggestion": UNDERFITTING_SUGGESTION})

    if run.learning_rate > HIGH_LEARNING_RATE and run.validation_accuracy < LOW_ACCURACY:
        recommendations.append({"issue": "Learning Rate too high", "suggestion": HIGH_LR_SUGGESTION})

    if run.learning_rate < LOW_LEARNING_RATE and run.validation_accuracy < LOW_ACCURACY:
        recommendations.append({"issue": "Learning Rate too low", "suggestion": LOW_LR_SUGGESTION})

    if run.epochs < MINIMUM_EPOCHS and run.validation_accuracy < LOW_ACCURACY:
        recommendations.append({"issue": "Too few epochs for model training", "suggestion":LOW_EPOCHS_SUGGESTION})

    if not recommendations:
        recommendations.append({"issue": "Healthy", "suggestion": "The model is healthy"})

    return recommendations


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
        if metric not in SORTABLE_METRICS:
            raise ValueError("Please enter either Validation Accuracy or Validation Loss")
        descending = True if metric == "validation_accuracy" else False
        runs = list(self.get_models().items())
        sorted_runs = merge_sort(items=runs,key=lambda x: getattr(x[1],metric),reverse=descending)
        return sorted_runs

    def filter_by_accuracy(self,threshold):
        if not isinstance(threshold,(float,int)) or not 0 <= threshold <= 100:
            raise ValueError("Please enter a valid accuracy threshold")
        filtered_runs = [(run_id,run) for run_id,run in self.models.items() if run.validation_accuracy >= threshold]
        return filtered_runs

    
    def filter_by_loss(self,threshold):
        if not isinstance(threshold,(float,int)) or not threshold >= 0:
            raise ValueError("Please enter a valid loss threshold")
        filtered_runs = [(run_id,run) for run_id,run in self.models.items() if run.validation_loss <= threshold]
        return filtered_runs

    def filter_by_name(self,name):
        if not isinstance(name,str) or not name.strip():
            raise ValueError("Please enter a valid model name")
        filtered_runs = [(run_id,run) for run_id,run in self.models.items() if name.strip().lower() in run.model_name.lower()]
        return filtered_runs

    def get_statistics(self,metric):
        if metric not in SORTABLE_METRICS:
            raise ValueError("Please enter a valid metric to compute statistics")
        if not self.models:
            raise ValueError("No runs exist to compute statistics on")
        values = [getattr(run,metric) for run in self.models.values()]
        sorted_runs = self.sort_runs(metric)
        best_run_id = sorted_runs[0][0]
        return {
            "mean": mean(values),
            "variance": variance(values),
            "std_dev": std_dev(values),
            "best_run_id": best_run_id
        }

    def recommend(self,run_id):
        if run_id not in self.models.keys():
            raise KeyError(f"A model with ID {run_id} doesn't exist")
        return get_recommendations(self.models[run_id])
