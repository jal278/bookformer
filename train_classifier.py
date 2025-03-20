import embed_reviews
from embed_reviews import get_embedding, remove_quotes
import pickle
import spacy
import random
from config import data_dir, base_dir, openai_api_key
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


nlp = spacy.load('en_core_web_sm')
model = "text-embedding-3-small"
m_abbrev = "3s"

# load one of the embedding files, to grab arbitrary negative examples from
emb_file = 'review-sent-embeddings-{}.pkl'.format(m_abbrev)
_emb = pickle.load(open(emb_file, 'rb'))
emb = _emb['emb']
sent_embeddings = emb

pos = []
neg = []

# file that has positive/negative examples in plain text
fname = "classifier_changed_my_life.txt"
#fname = "classifier_worst_book.txt"
#fname = "classifier_gift.txt"
#fname = "classifier_assigned.txt"
#fname = 'classifier_digust.txt'
#fname = 'classifier_perma_p.txt'
#fname = 'classifier_perma_e.txt'
#fname = 'classifier-surprise.txt'
#fname = 'classifier_erotic.txt'
#fname = "classifier_best.txt"
#fname = "classifier_weird.txt"

lines = open("classifiers/"+fname,"r").read().split("\n")[:-1]
# map all the text examples in the file to embeddings (the classifier works on embeddings)
for line in lines:
    line = line.strip()
    if line[0]=='-':
        neg.append(get_embedding(line[1:]))
    if line[0]=='+':
        pos.append(get_embedding(line[1:]))


# to augment the positives and hard negatives (from the text file), we also add random negative examples

# grab 80 random negative examples
neg_examples = random.sample(list(sent_embeddings.values()), 80)
# randomly sample one sentence from each negative example
neg_examples = [np.array(random.choice(x)) for x in neg_examples]
# add the negative examples to the list
neg += neg_examples

# split into data (X) and labels (y)
X = pos+neg
y = [1]*len(pos) + [0]*len(neg)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)

#clf = MLPClassifier(random_state=1, max_iter=1000,alpha=1).fit(X_train, y_train)
#clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
#clf = RandomForestClassifier(max_depth=5, n_estimators=10).fit(X_train, y_train)
#clf = SVC(gamma=2,C=1, probability=True).fit(X_train, y_train)
#clf = AdaBoostClassifier(algorithm='SAMME').fit(X_train, y_train)

# logistic regression is easy/reliable to train and works well for this task
# could experiment with other classifiers if we want to
clf = LogisticRegression().fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)

# save the classifier
pickle.dump(clf, open('classifier_{}.pkl'.format(fname), 'wb'))

