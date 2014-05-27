from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.stem.lancaster import LancasterStemmer
import csv

def split_and_stem_words(s):
    '''
    Split a sentence to a list of words, and then do stemming for each word.
    '''
    st = LancasterStemmer()
    # The best!!!
    return map(st.stem, [w.strip().strip('\\\n\\r') for w in s.split(' ')])
    #return [w.strip().strip('\\n\\r\\n') for w in s.split(' ')]
    #return PunktWordTokenizer().tokenize(s)

def parse_essay(path_essay):
    '''
    Parse the file to a list [(projectID: string, teacherID: string, title: [word: string]), short_introduction: [word: string], need: [word: string], essay: [word: string]]
    '''
    data_essay = []
    csv_reader_essay = csv.reader(open(path_essay), delimiter=',', quotechar='"')
    n_row = -1
    for row in csv_reader_essay:
        # Format of row: projectID, teacherID, title, short introduction, needs, essay
        n_row += 1
        if n_row == 0:
            continue
        data_essay.append(row[:2] + map(split_and_stem_words, row[2: ]))
    return data_essay

def gen_data_target():
    '''
    Generate feature matrix and target vectors for sklearn
    '''
    #path_essay = "../Data/essays.csv"
    path_essay = "/Users/jwpan/Work/Labs/KDD_CUP_2014/Data/essays_1000.csv"
    data_essay = parse_essay(path_essay)
    #path_essay_processed = path_essay.split('.')[0] + '_processed_nltk_PunktWordTokenizer.csv'
    path_essay_processed = path_essay.split('.')[0] + '_processed_strip_split_M_stem.csv'
    file_essay_processed = open(path_essay_processed, 'w')
    for row in data_essay:
        file_essay_processed.write(str(row) + '\n')
    file_essay_processed.close()
    
def Naive_Bayes_Example():
    '''
    The Baive Bayes Classifier provided by Sklearn
    '''
    iris = datasets.load_iris()
    print 'iris.data', iris.data
    print 'iris.target', iris.target
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print ("Number of mislabeled points: %d"%(iris.target != y_pred).sum())

if __name__=='__main__':
    #Naive_Bayes_Example()
    gen_data_target()
    #verify_delimeter()
