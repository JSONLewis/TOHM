from tohm import Container
#from CatsVsDogs import net
from Lean import TextSentiment
from Lean import Net
import torch
import numpy as np 
import cv2
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTLABELS = {0:"CAT",1:"DOG"} 
ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}
try: 
    img = cv2.imread("puppy.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(50,50))
except Exception as e:
    pass     
testimage =torch.Tensor([np.array(img)]).view(-1,1,50,50) 

checkpoint = torch.load('textsentiment.pt') 
VOCAB_SIZE = checkpoint['vocab_size']
EMBED_DIM = checkpoint['embed_dim']
NUN_CLASS = checkpoint['nun_class']
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device) 
model.load_state_dict(checkpoint['model_state_dict']) 
model.eval()
model.vocab = checkpoint['vocab']
model.ngrams = 2
ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."
model.text = ex_text_str
container = Container()


tokenizer = get_tokenizer("basic_english")
with torch.no_grad():
    text = torch.tensor([model.vocab[token]
    for token in ngrams_iterator(tokenizer(model.text), model.ngrams)])
model.tdog = torch.tensor([0])

checkpoint2 = torch.load('catsvsdogs.pt') 
net = Net().to(device) 
net.load_state_dict(checkpoint2['model_state_dict'])
net.eval()
container.addmodel(net,RESULTLABELS, "Model1234")
container.addmodel(model,ag_news_label)

result = container.predict(testimage)
print(container.model_used)
print(container.label_used)
result2 = container.predict(text)
if result is not None:
    print(RESULTLABELS[result])
    print(ag_news_label[result2 +1])#TODO: why plus one (must be zero based)
    print(container.model_used)
    print(container.label_used)
