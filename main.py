import random
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.classify.util import accuracy

# "Stop words" that you might want to use in your project/an extension
stop_words = set(stopwords.words('english'))

def format_sentence(sent):
    ''' format the text setence as a bag of words for use in nltk'''
    tokens = nltk.word_tokenize(sent)
    return({word: True for word in tokens})

def get_reviews(data, rating):
    ''' Return the reviews from the rows in the data set with the
        given rating '''
    rows = data['Rating']==rating
    return list(data.loc[rows, 'Review'])


def split_train_test(data, train_prop):
    ''' input: A list of data, train_prop is a number between 0 and 1
              specifying the proportion of data in the training set.
        output: A tuple of two lists, (training, testing)
    '''
    # TODO: You will write this function, and change the return value
    return ([], [])


'''
Markov Chain below
'''
'''
Example reading. 
f = open("file.txt")
text = f.read()
s = text.split()
'''
# train the markov chain on a string
def train(s):
  dictionary = dict() #makes a blank dictionary
  words = s.split() #split string into a list of words
  # for each word, get its index. if not in dictionary, add a key-value pair of the word and [the word after it]. if in dictionary, add the next word to the [word] list that's its value
  for index in range(len(words)):
    word = words[index]
    # if last word, wrap around
    if index < int(len(words)-1):
      if word not in dictionary:
        dictionary[word] = []
        dictionary[word].append(words[index+1])
      else:
        dictionary[word].append(words[index+1])
    else:
      if word not in dictionary:
        dictionary[word] = []
        dictionary[word].append(words[0])
      else:
        dictionary[word].append(words[0])
  return dictionary
# # test cases for train(s)
# train("Six elephants sit quietly, too, but rise; they like apples and like yogurt too and other stuff, too. Why? I don't know!")
# print(train("Bonjour. C'est moi, Richard, et je suis un étudiant à l'Université de Californie à San Diego. J'ai 18 ans, et j'aime les maths et l'informatique."))
# model = dictionary; start_word = starting word; num_words = number of words

  
def generate(model, num_sentences, first_word):
  wordLimit = 0
  sentences = 0
  dictionary = train(model)
  # firstList = startWordList(model)
  # ranNum = random.randrange(0, len(firstList))
  currentWord = str(first_word)
  sentence = ""
  while sentences < num_sentences:
    sentence = sentence + " " + currentWord
    ranNum = random.randrange(0, len(dictionary[currentWord]))
    currentWord = dictionary[currentWord][ranNum]
    wordLimit += 1
    if wordLimit >= 200:
      break
    if currentWord[len(currentWord)-1] == "." or currentWord[len(currentWord)-1] == "!" or currentWord[len(currentWord)-1] == "?":
      sentences += 1
  return sentence
# if you want to just choose a random first word, use this instead in the function rather than first_word
def startWordList(s):
  words = s.split()
  wordList = []
  for w in words:
    if w[0].isupper():
      wordList.append(w)
  return wordList


copypasta = open("data/atestthatbreaks.txt").read()
print(generate(copypasta, 2, "i"))
'''
Markov Chain above
'''


# def format_for_classifier(data_list, label):
#     ''' input: A list of documents represented as text strings
#                The label of the text strings.
#         output: a list with one element for each doc in data_list,
#                 where each entry is a list of two elements:
#                 [format_sentence(doc), label]
#     '''
#     # TODO: Write this function, change the return value
#     return []

# def classify_reviews():
#     ''' Perform sentiment classification on movie reviews ''' 
#     # Read the data from the file
#     data = pd.read_csv("data/movie_reviews.csv")

#     # get the text of the positive and negative reviews only.
#     # positive and negative will be lists of strings
#     # For now we use only very positive and very negative reviews.
#     positive = get_reviews(data, 4)
#     negative = get_reviews(data, 0)

#     # Split each data set into training and testing sets.
#     # You have to write the function split_train_test
#     (pos_train_text, pos_test_text) = split_train_test(positive, 0.8)
#     (neg_train_text, neg_test_text) = split_train_test(negative, 0.8)

#     # Format the data to be passed to the classifier.
#     # You have to write the format_for_classifier function
#     pos_train = format_for_classifier(pos_train_text, 'pos')
#     neg_train = format_for_classifier(neg_train_text, 'neg')

#     # Create the training set by appending the pos and neg training examples
#     training = pos_train + neg_train

#     # Format the testing data for use with the classifier
#     pos_test = format_for_classifier(pos_test_text, 'pos')
#     neg_test = format_for_classifier(neg_test_text, 'neg')
#     # Create the test set
#     test = pos_test + neg_test


#     # Train a Naive Bayes Classifier
#     # Uncomment the next line once the code above is working
#     #classifier = NaiveBayesClassifier.train(training)

#     # Uncomment the next two lines once everything above is working
#     #print("Accuracy of the classifier is: " + str(accuracy(classifier, test)))
#     #classifier.show_most_informative_features()

#     # TODO: Calculate and print the accuracy on the positive and negative
#     # documents separately
#     # You will want to use the function classifier.classify, which takes
#     # a document formatted for the classifier and returns the classification
#     # of that document ("pos" or "neg").  For example:
#     # classifier.classify(format_sentence("I love this movie. It was great!"))
#     # will (hopefully!) return "pos"

#     # TODO: Print the misclassified examples


# if __name__ == "__main__":
#     classify_reviews()
