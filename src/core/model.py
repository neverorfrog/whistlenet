from abc import ABC, abstractmethod
import os
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import torch
from torch import nn
from core.utils import Parameters
from core.utils import project_root

projroot = project_root()
root = f"{projroot}/model-weights/torch"

class Model(nn.Module, Parameters, ABC):
    """The base class of models"""
    def __init__(self, name=None):
        super().__init__()
        self.save_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, X):
        pass
    
    def save(self):
        if self.name is None: return # TODO
        path = os.path.join("models",self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.state_dict(), open(os.path.join(path,"model.pt"), "wb"))
        print("MODEL SAVED!")

    def load(self, name):
        if self.name is None: return # TODO
        path = os.path.join("models",name)
        self.load_state_dict(torch.load(open(os.path.join(path,"model.pt"),"rb")))
        self.eval()
        print("MODEL LOADED!")
            
class Classifier(Model):
    """The base class of models. Not instantiable because forward inference has to be defined by subclasses."""
    def __init__(self, name, num_classes, bias=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_scores = []
        self.train_scores = []
        self.val_scores = []
        self.training_time = 0
        self.save_parameters() #saves as class fields the parameters of the constructor
        
    def training_step(self,batch): #forward propagation
        inputs = batch[:-1][0].type(torch.float).to(self.device) #one sample on each row -> X.shape = (m, d_in)
        # inputs = inputs.reshape(-1,1,513) # TODO hardcodato
        inputs = inputs.reshape(-1,1,28,28) # TODO hardcodato
        labels = batch[-1].type(torch.long) # labels -> shape = (m)
        logits = self(inputs).squeeze()
        loss = self.loss_function(logits, labels)
        return loss
    
    def validation_step(self,batch):
        with torch.no_grad():
            inputs = batch[:-1][0].type(torch.float).to(self.device) #one sample on each row -> X.shape = (m, d_in)
            # inputs = inputs.reshape(-1,1,513) # TODO hardcodato
            inputs = inputs.reshape(-1,1,28,28) # TODO hardcodato
            labels = batch[-1].type(torch.long) # labels -> shape = (m)
            logits = self(inputs).squeeze()
            loss = self.loss_function(logits, labels)
            predictions = logits.argmax(axis = 1).squeeze().detach().type(torch.long).to(self.device) # the most probable class is the one with highest probability
            report = classification_report(batch[-1],predictions, output_dict=True)
            score = report['weighted avg']['f1-score']
        return loss, score
    
    @property
    @abstractmethod
    def loss_function(self):
        """
        A getter method for the loss function property.
        """
        pass
        
    @abstractmethod
    def forward(self, X):
        """
        A method to perform the forward pass using the given input data X.
        """
        pass

    def predict(self, inputs):
        """
        A method to make predictions using the input data X.
        """
        with torch.no_grad():
            return torch.softmax(self(inputs), dim=-1).argmax(axis = -1).squeeze() #shape = (m)
    
    def save(self):
        path = os.path.join(root,self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.state_dict(), open(os.path.join(path,"model.pt"), "wb"))
        torch.save(self.test_scores, open(os.path.join(path,"test_scores.pt"), "wb")) 
        torch.save(self.train_scores, open(os.path.join(path,"train_scores.pt"), "wb"))
        torch.save(self.val_scores, open(os.path.join(path,"val_scores.pt"), "wb"))
        torch.save(self.training_time, open(os.path.join(path,"training_time.pt"), "wb"))
        print("MODEL SAVED!")

    def load(self, name):
        path = os.path.join(root,name)
        self.load_state_dict(torch.load(open(os.path.join(path,"model.pt"),"rb")))
        self.test_scores = torch.load(open(os.path.join(path,"test_scores.pt"),"rb"))
        self.train_scores = torch.load(open(os.path.join(path,"train_scores.pt"),"rb"))
        self.val_scores = torch.load(open(os.path.join(path,"val_scores.pt"),"rb"))
        self.training_time = torch.load(open(os.path.join(path,"training_time.pt"),"rb"))
        self.eval()
        print("MODEL LOADED!")
        
    def plot(self, name, complete=False):        
        plt.plot(self.test_scores, label=f'{name} - test scores')
        if complete:
            plt.plot(self.train_scores, label=f'{name} - train scores')
            plt.plot(self.val_scores, label=f'{name} - val scores')
        plt.legend()
        plt.ylabel('score')
        plt.xlabel('epoch')
        plt.show()
        
    def evaluate(self,data,show=True):
        test_dataloader = data.test_dataloader(len(data.test_data))
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch[:-1][0].detach().type(torch.float).to(self.device) #one sample on each row -> X.shape = (m, d_in)
                inputs = inputs.reshape(-1,1,28,28) # TODO hardcodato
                labels = batch[-1].detach().type(torch.long).to(self.device)
                predictions_test = self.predict(inputs)
                report_test = classification_report(labels, predictions_test, digits=3, output_dict=True)
                if show:
                    print(report_test)
                self.test_scores.append(report_test['accuracy']) # TODO hardcodato