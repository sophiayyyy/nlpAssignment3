import glob
import torch
import unicodedata
import string
import random
import time
import math
from itertools import product
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.autograd import Variable

category_lines_val = {}
all_val_categories = []

category_lines_train = {}
all_train_categories = []

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    combined = torch.cat((input, hidden), 1)
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    output = self.softmax(output)
    return output, hidden

  def init_hidden(self):
    return Variable(torch.zeros(1, self.hidden_size))

def unicode_to_ascii(filename):
  with open(filename, encoding="ISO-8859-1") as f:
    vocab =  [line.strip() for line in f]
  #print ("length of vocab is:", len(vocab))
  all_letters = string.ascii_letters + " .,;'"
  #n_letters = len(all_letters)
  ascii_vocab = list()
  for v in vocab:
    s =  ''.join(
      c for c in unicodedata.normalize('NFD', v)
      if unicodedata.category(c) != 'Mn'
      and c in all_letters
    )
    ascii_vocab.append(s)
  #print ("list len is ", len(ascii_vocab))
  #print ("list is ", ascii_vocab)
  return ascii_vocab

def input():
  input = Variable(line_to_tensor('Albert'))
  hidden = Variable(torch.zeros(1, n_hidden))


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
  all_letters = string.ascii_letters + " .,;'"
  n_letters = len(all_letters)
  tensor = torch.zeros(len(line), 1, n_letters)
  for li, letter in enumerate(line):
      letter_index = all_letters.find(letter)
      tensor[li][0][letter_index] = 1
  return tensor

def test_rnn(rnn):
  input = Variable(line_to_tensor('Albert'))
  hidden = Variable(torch.zeros(1, n_hidden))

  output, next_hidden = rnn(input[0], hidden)
  print(output)

def train_category_from_output(output):
  top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
  category_i = top_i[0][0]
  return all_train_categories[category_i], category_i

def val_category_from_output(output):
  top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
  category_i = top_i[0][0]
  return all_val_categories[category_i], category_i

def random_training_pair(all_categories, category_lines):
  category = random.choice(all_categories)
  line = random.choice(category_lines[category])
  category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
  line_tensor = Variable(line_to_tensor(line))
  return category, line, category_tensor, line_tensor

def random_val_pair(all_categories, category_lines):
  category = random.choice(all_categories)
  line = random.choice(category_lines[category])
  category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
  line_tensor = Variable(line_to_tensor(line))
  return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
  rnn.zero_grad()
  hidden = rnn.init_hidden()

  for i in range(line_tensor.size()[0]):
    output, hidden = rnn(line_tensor[i], hidden)

  loss = criterion(output, category_tensor)
  loss.backward()

  optimizer.step()

  return output, loss.data#[0]

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Just return an output given a line
def evaluate(category_tensor,line_tensor):
  hidden = rnn.init_hidden()

  for i in range(line_tensor.size()[0]):
    output, hidden = rnn(line_tensor[i], hidden)

  loss = criterion(output, category_tensor)
  return output, loss.data

if __name__ == '__main__':
  all_letters = string.ascii_letters + " .,;'"
  n_letters = len(all_letters)

  all_train_filenames = glob.glob('train/*.txt')
  #print ("train data names: ", all_train_filenames)
  # category_lines_train = {}
  # all_train_categories = []
  for filename in all_train_filenames:
    # Turn a Unicode string to plain ASCII,  thanks to http://stackoverflow.com/a/518232/2809427
    ascii_vocab = unicode_to_ascii(filename)
    # get category
    category = filename.split('/')[-1].split('.')[0]
    #print (category)
    all_train_categories.append(category)
    category_lines_train[category] = ascii_vocab
    # print ("train category", category)
    # print ("len: ", len(category_lines_train[category]))
  n_train_categories = len(all_train_categories)
  #print('n_train_categories =', n_train_categories)
  #print(line_to_tensor('Jones').size())

  all_val_filenames = glob.glob('val/*.txt')
  #print ("val data names: ", all_val_filenames)
  # category_lines_val = {}
  # all_val_categories = []
  for filename in all_val_filenames:
    # Turn a Unicode string to plain ASCII,  thanks to http://stackoverflow.com/a/518232/2809427
    ascii_vocab = unicode_to_ascii(filename)
    # get category
    category = filename.split('/')[-1].split('.')[0]
    #print (category)
    all_val_categories.append(category)
    category_lines_val[category] = ascii_vocab
    # print ("val category", category)
    # print ("len: ", len(category_lines_val[category]))
  n_val_categories = len(all_val_categories)
  #print('n_val_categories =', n_val_categories)
  #print(line_to_tensor('Jones').size())


  n_hiddens = [100]
  learning_rates = [0.0015]
  n_iters = [150000]
  for n_hidden, learning_rate, n_iter in product(n_hiddens, learning_rates, n_iters):
    #n_hidden = 128
    print ("n_hiddens is: ", n_hidden)
    print ("learning_rate is: ", learning_rate)
    print ("n_iter is: ", n_iter)
    rnn = RNN(n_letters, n_hidden, n_train_categories)
    #test_rnn(rnn)

    criterion = nn.NLLLoss()
    #learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    n_epochs = n_iter
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_train_loss = 0
    current_val_loss = 0
    all_train_losses = []
    all_val_losses = []
    val_loss = 0
    start = time.time()

    for epoch in range(1, n_epochs + 1):
      # Get a random training input and target
      category, line, category_tensor, line_tensor = random_training_pair(all_train_categories, category_lines_train)
      train_output, loss = train(category_tensor, line_tensor)
      current_train_loss += loss

      category, line, category_tensor, line_tensor = random_val_pair(all_val_categories, category_lines_val)
      val_output,val_loss = evaluate(category_tensor, line_tensor)
      current_val_loss += val_loss


      # Print epoch number, loss, name and guess
      if epoch % print_every == 0:
        guess, guess_i = train_category_from_output(train_output)
        if guess == category:
          correct = 'Correct'
        else:
          correct = 'Incorrect ' + '(' + category + ')'#% category
        print(
        '%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

      # Add current loss avg to list of losses
      if epoch % plot_every == 0:
        all_train_losses.append(current_train_loss / plot_every)
        current_train_loss = 0
        all_val_losses.append(current_val_loss / plot_every)
        current_val_loss = 0

    #get loss result
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(all_train_losses, label = "training_loss")
    plt.plot(all_val_losses, label = "validation_loss")
    lossname = "loss_" + str(n_hidden) + "_" + str(learning_rate) + "_" + str(n_iter) + ".png"
    plt.legend()
    plt.savefig(lossname)


    #evaluate

    # Go through a bunch of examples and record to test if it is overfitted
    test = 100000
    train_correct_sum = 0.0
    macro_train = 0.0
    micro_train = 0.0
    tmp = 0.0
    train_correct = torch.zeros(n_train_categories)
    train_incorrect = torch.zeros(n_train_categories)
    for i in range(test):
      category, line, category_tensor, line_tensor = random_training_pair(all_train_categories, category_lines_train)
      output, loss = evaluate(category_tensor, line_tensor)
      guess, guess_i = train_category_from_output(output)
      category_i = all_train_categories.index(category)
      if category_i == guess_i:
        train_correct_sum += 1
        train_correct[category_i] += 1
      else:
        train_incorrect[category_i] += 1
    micro_train = train_correct_sum / test
    for index in range(n_train_categories):
      tmp += train_correct[index] / (train_correct[index] + train_incorrect[index])
    macro_train = tmp / n_train_categories
    print ("Macro_train accuracy is: ", macro_train)
    print ("Micro_train accuracy is: ", micro_train)

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_val_categories, n_val_categories)
    n_confusion = 100000

    # Go through a bunch of examples and record which are correctly guessed
    val_correct_sum = 0.0
    macro_val = 0.0
    micro_val = 0.0
    tmp = 0.0
    val_correct = torch.zeros(n_val_categories)
    val_incorrect = torch.zeros(n_val_categories)
    # for i in range(n_confusion):
    #   category, line, category_tensor, line_tensor = random_val_pair(all_val_categories, category_lines_val)
    #   output, loss = evaluate(category_tensor, line_tensor)
    #   guess, guess_i = val_category_from_output(output)
    #   category_i = all_val_categories.index(category)
    #   confusion[category_i][guess_i] += 1
    #   if category_i == guess_i:
    #     val_correct_sum += 1
    #     val_correct[category_i] += 1
    #   else:
    #     val_incorrect[category_i] += 1
    # micro_val = val_correct_sum / n_confusion
    # for index in range(n_val_categories):
    #   tmp += val_correct[index] / (val_correct[index] + val_incorrect[index])
    # macro_val = tmp / n_val_categories
    # print ("Macro_val accuracy is: ", macro_val)
    # print ("Micro_val accuracy is: ", micro_val)
    num_confusion = 0
    for category in all_val_categories:
      num_confusion += len(category_lines_val[category])
      #print (num_confusion)
      for line in category_lines_val[category]:
        category_tensor = Variable(torch.LongTensor([all_val_categories.index(category)]))
        line_tensor = Variable(line_to_tensor(line))
        output, loss = evaluate(category_tensor, line_tensor)
        guess, guess_i = val_category_from_output(output)
        category_i = all_val_categories.index(category)
        confusion[category_i][guess_i] += 1
        if category_i == guess_i:
          val_correct_sum += 1
          val_correct[category_i] += 1
        else:
          val_incorrect[category_i] += 1
    print ("num_confusion is: ", num_confusion)
    micro_val = val_correct_sum / num_confusion
    for index in range(n_val_categories):
      tmp += val_correct[index] / (val_correct[index] + val_incorrect[index])
    macro_val = tmp / n_val_categories
    print ("Macro_val accuracy is: ", macro_val)
    print ("Micro_val accuracy is: ", micro_val)

    # Normalize by dividing every row by its sum
    for i in range(n_val_categories):
      confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_val_categories, rotation=90)
    ax.set_yticklabels([''] + all_val_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #plt.show()
    evaname = "evaluate_" + str(n_hidden) + "_" + str(learning_rate) + "_" + str(n_iter) + ".png"
    plt.savefig(evaname)


