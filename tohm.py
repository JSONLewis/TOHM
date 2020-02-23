 
class Container() :
    def __init__(self):
        self.models = []
        self.signatures = {}
        self.labels = {}

 
    def addmodel(self, model, resultlabels, identifier="blah"):
        #type check model 
        self.models.append(model)
        # Add identifier to Model
        model.identifier = identifier #add guid etc
        self.labels[model.identifier] = resultlabels
        #print(self.models)

    def predict(self, tensor):
        for model in self.models:
            #Build caches dictionary
            tensor.shape
            tensor.size 
            if model.identifier in self.signatures:
                if self.signatures[model.identifier] != tensor.size :
                    continue 
            try:
                output = model( tensor)#, model.tdog
                self.signatures[model.identifier] = tensor.size   
                #print(resultlabels[output.argmax(1).item()])
                self.label_used = self.labels[model.identifier]
                self.model_used = model.identifier 
                return output.argmax(1).item()
            except Exception as e:
                pass

              
