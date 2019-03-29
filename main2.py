import unidecode
import string
import random
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

chunk_len = 200
all_characters = string.printable
n_characters = len(all_characters)

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers=1):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers

    self.encoder = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
    self.decoder = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden):
    input = self.encoder(input.view(1, -1))
    output, hidden = self.gru(input.view(1, 1, -1), hidden)
    output = self.decoder(output.view(1, -1))
    return output, hidden

  def init_hidden(self):
    return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


def evaluate(prime_str='A', predict_len=1, temperature=1):
  hidden = decoder.init_hidden()
  prime_input = char_tensor(prime_str)
  predicted = prime_str

  # Use priming string to "build up" hidden state
  for p in range(len(prime_str) - 1):
    _, hidden = decoder(prime_input[p], hidden)
  inp = prime_input[-1]

  for p in range(predict_len):
    output, hidden = decoder(inp, hidden)

    # Sample from the network as a multinomial distribution
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1)[0]
    #print ("torch.multinomial(output_dist, 1)", torch.multinomial(output_dist, 1))

    # Add predicted character to string and use as next input
    predicted_char = all_characters[top_i]
    predicted += predicted_char
    inp = char_tensor(predicted_char)

  return predicted

def evaluate_new(prime_str, predict_len, res_str, temperature):
  hidden = decoder.init_hidden()
  prime_input = char_tensor(prime_str)
  predicted = prime_str

  # Use priming string to "build up" hidden state
  for p in range(len(prime_str) - 1):
    _, hidden = decoder(prime_input[p], hidden)
  inp = prime_input[-1]

  for p in range(predict_len):
    output, hidden = decoder(inp, hidden)

    # Sample from the network as a multinomial distribution
  output_dist = output.data.view(-1).div(temperature).exp()
  car_idx = char_tensor(res_str)
  #print ('car_idx is ', car_idx)
  preplexity = math.log(output_dist[car_idx] / output_dist.sum())

    # Add predicted character to string and use as next input
    # predicted_char = all_characters[top_i]
    # predicted += predicted_char
    # inp = char_tensor(predicted_char)

  return preplexity

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / chunk_len

if __name__ == '__main__':
  file = unidecode.unidecode(open('speeches.txt').read())
  file_len = len(file)
  print('file_len =', file_len)

  #print (random_chunk())

  #print(char_tensor('abcDEF'))
  n_epochs = 2000
  print_every = 100
  plot_every = 10
  hidden_size = 100
  n_layers = 1
  lr = 0.005

  decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  start = time.time()
  all_losses = []
  loss_avg = 0

  for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())
    loss_avg += loss

    if epoch % print_every == 0:
      print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
      print(evaluate('Wh', 100), '\n')

    if epoch % plot_every == 0:
      all_losses.append(loss_avg / plot_every)
      loss_avg = 0

  plt.figure()
  plt.plot(all_losses)
  plt.savefig("loss.png")

  val_string = ['In other words, the United States is stronger, safer, and a richer country than it was when I assumed office less than two years ago.',
                   'We are standing up for America and for the American people. And we are also standing up for the world.',
                   'With support from many countries here today, we have engaged with North Korea to replace the specter of conflict with a bold and new push for peace.',
                   'Ultimately, it is up to the nations of the region to decide what kind of future they want for themselves and their children.',
                   'Every solution to the humanitarian crisis in Syria must also include a strategy to address the brutal regime that has fueled and financed it: the corrupt dictatorship in Iran.',
                   'But the important point I want to make here is that we already are deeply engaged in trying to bring about a solution in Syria.',
                   'For all the maps plastered across our TV screens today, and for all the cynics who say otherwise, I continue to believe we are simply more than just a collection of red and blue states.  We are the United States.',
                   'Now, I am not running for office anymore, so let me just present the facts.  I promised that 2014 would be a breakthrough year for America.  This morning, we got more evidence to back that up.  In December, our businesses created 240,000 new jobs.',
                   'I think we should have further broad-based debate among the American people.  As I have said before, I do think that the episode with the unaccompanied children changed a lot of attitudes.',
                   'And we stand for freedom and hope and the dignity of all human beings.  And that is what the city of Paris represents to the world, and that spirit will endure forever -- long after the scourge of terrorism is banished from this world.',
                   'really like their strong coffee But its more than that. There is the whole experience First I have to deal with that line up.Depending on the location and time of day Your gonna find about 6 to 8 people in that line up.',
                   'I never even recognize when I look like someone. I will be out somewhere and be all like....hahaha! Look at that guy, who does that dork look like?" and then sadly realize... "OH MY GOSH!...that freak looks like me!" eh....what are you gonna do?',
                   'This skit is also known as "THE GOLF PARADOX" - enjoy:I dont understand golf.The entire point of golf is to play as little golf as possible.To make as few shots as possible.',
                   'I am going straight to the top you guys. I got my first gig lined up this weekend. Come out if you can. Its the Peachtree 20__ talent showcase. Its gonna be INSANE *pause* in the membrane!My mom, i mean my manager Sheila said i got some real fresh talent!But my career as a rapper really depends on you guys, the fans.',
                   'None of this would be possible without you guys! and I really need you guys to spread the word.I have a lot of catching up to do since my former partner, marshall stole my song rap god.Until next time, stay wicked!']
  print (val_string)
  res_str = ' '
  perplexity = 0.0
  for i in range(len(val_string)):
    print ("No.", i + 1)
    perplexity = 0.0
    res_str = ' '
    for index in range(len(val_string[i])):
      perplexity += evaluate_new(res_str, 1, val_string[i][index], 1)
      res_str += val_string[i][index]
    tmp = math.exp((-1) * (1/float(len(val_string[i]))) * perplexity)
    print ("the perplexity is: ", tmp)