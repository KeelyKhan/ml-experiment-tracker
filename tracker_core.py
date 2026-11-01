from dataclasses import dataclass

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

