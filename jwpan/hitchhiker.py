from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.stem.lancaster import LancasterStemmer
import csv, math, random

def gen_projectID(path, flag):
    path_res = '../Data/projectID_' + flag + '.txt'
    file_res = open(path_res, 'w')
    n_line = -1
    for line in open(path):
        n_line += 1
        if n_line == 0:
            continue
        row = line.strip('\n').split('\t')
        file_res.write(row[0] + '\n')
    file_res.close()

def gen_train_validation_test_projectID():
    '''
    JUST IGNORE THIS FUNCTION!!
    Gen train/validation/test projectID from the first gbdt fv data
    '''
    path_train_fv = '../Data/data_FB_gbdt_format_train.fv'
    path_validation_fv = '../Data/data_FB_gbdt_format_validation.fv'
    path_test_fv = '../Data/data_FB_gbdt_format_test.fv'
    gen_projectID(path_train_fv, 'train')
    gen_projectID(path_validation_fv, 'validation')
    gen_projectID(path_test_fv, 'test')

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
        if n_row % 6641 == 1:
            print n_row / 664099.0
        if n_row == 0:
            continue
        data_essay.append(row[:2] + map(split_and_stem_words, row[2: ]))
    return data_essay

def gen_data_target():
    '''
    Generate feature matrix and target vectors for sklearn
    '''
    #path_essay = "../Data/essays.csv"
    path_essay = "/Users/jwpan/Work/Labs/KDD_CUP_2014/Data/essays.csv"
    data_essay = parse_essay(path_essay)
    #path_essay_processed = path_essay.split('.')[0] + '_processed_nltk_PunktWordTokenizer.csv'
    path_essay_processed = path_essay.split('.')[0] + '_split_and_stemming.txt'
    file_essay_processed = open(path_essay_processed, 'w')
    for row in data_essay:
        #file_essay_processed.write(str(row) + '\n')
        file_essay_processed.write(''.join(row[:2]) + '' + ''.join([''.join(r) for r in row[2:]]) + '\n')
    file_essay_processed.close()

def gen_feedback_feature():
    '''
    Generate feedback feature from project.csv
    '''
    path_project = '../Data/projects.csv'
    path_project_train = '../Data/projects_joined_train.csv'
    path_outcome = '../Data/outcomes.csv'
    # projectID as key, outcome as value: 1 for exiting and 0 for unexiting
    d_outcome = {}
    index_project_discrete_feature = [6, 7] + range(9, 28) + [32, 33]
    print 'Begin build d_outcome'
    for line in open(path_outcome):
        row = line.strip('\n').split(',')
        value = -1
        if row[1] == 't':
            value = 1
        elif row[1] == 'f':
            value = 0
        d_outcome.setdefault(row[0], value)
    print 'Finish build d_outcome'
    # Store the number of exiting/unexiting projects for each discrete value of each feature
    d_feature = {}
    # Store all discrete feature type in project.csv
    feature_type_project = []
    n_line = -1
    file_project = open(path_project_train)
    print 'Begin build d_feature'
    for line in file_project:
        n_line += 1
        if n_line % 6641 == 6640:
            print n_line / 664099.0
        row = line.strip('\n').split(',')
        if n_line == 0:
            #feature_type_project = [row[i] for i in index_project_discrete_feature]
            feature_type_project = row
            continue
        projectID = row[0]
        for i in index_project_discrete_feature:
            feature_name = feature_type_project[i] + ':' + row[i]
            d_feature.setdefault(feature_name, {})
            d_feature[feature_name].setdefault('exiting', 0.0)
            d_feature[feature_name].setdefault('unexiting', 0.0)
            # Only training projects has outcome
            if d_outcome.has_key(projectID):
                if d_outcome[projectID] == 1:
                    d_feature[feature_name]['exiting'] += 1
                elif d_outcome[projectID] == 0:
                    d_feature[feature_name]['unexiting'] += 1
    print d_feature
    file_project.close()
    print 'Finish build d_feature'
    n_line = -1
    path_res = '../Data/data_FB_gbdt_format.fv'
    file_project = open(path_project)
    file_res = open(path_res, 'w')
    print 'Begin output gbdt format data'
    for line in file_project:
        n_line += 1
        if n_line % 6641 == 6640:
            print n_line / 664099.0
        row = line.strip('\n').split(',')
        if n_line == 0:
            discrete_feature_type_project = ['projectID'] + [row[i] for i in index_project_discrete_feature] + ['label']
            file_res.write('\t'.join(discrete_feature_type_project) + '\n')
            continue
        features = [row[0]]
        for i in index_project_discrete_feature:
            feature_name = feature_type_project[i] + ':' + row[i]
            if d_feature.has_key(feature_name):
                value = 0.0
                Sum_exiting_unexiting = d_feature[feature_name]['exiting'] + d_feature[feature_name]['unexiting']
                if Sum_exiting_unexiting > 0:
                    value = d_feature[feature_name]['exiting'] / Sum_exiting_unexiting
                features.append(str(value))
            else:
                features.append('0.0')
        if d_outcome.has_key(row[0]):
            if d_outcome[row[0]] == 1:
                features.append('1')
            elif d_outcome[row[0]] == 0:
                features.append('-1')
        else:
            features.append('unknown')
        file_res.write('\t'.join(features) + '\n')
    file_res.close()
    file_project.close()
    print 'End output gbdt format data'


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

def split_train_validation_test():
    path_data = '/Users/jwpan/Work/Labs/KDD_CUP_2014/Data/data_FB_gbdt_format.fv'
    path_train = path_data.split('.')[0] + '_train.fv'
    path_validation = path_data.split('.')[0] + '_validation.fv'
    path_test = path_data.split('.')[0] + '_test.fv'
    projectID_train = '../Data/projectID_train.txt'
    projectID_validation = '../Data/projectID_validation.txt'
    projectID_test = '../Data/projectID_test.txt'
    join_data_with_projectID(path_data, projectID_train, path_train)
    join_data_with_projectID(path_data, projectID_validation, path_validation)
    join_data_with_projectID(path_data, projectID_test, path_test)

def join_data_with_projectID(path_data, path_projectID, path_res):
    l_projectID = []
    for line in open(path_projectID):
        l_projectID.append(line.strip('\n'))
    s_projectID = set(l_projectID)
    file_res = open(path_res, 'w')
    n_line = -1
    for line in open(path_data):
        n_line += 1
        if n_line == 0:
            file_res.write(line)
            continue
        row = line.strip('\n').split('\t')
        if row[0] in s_projectID:
            file_res.write(line)
    file_res.close()

if __name__=='__main__':
    #Naive_Bayes_Example()
    #gen_data_target()
    #gen_feedback_feature()
    split_train_validation_test()
    #gen_train_validation_test_projectID()
    '''
    path_data = '/Users/jwpan/Work/Labs/KDD_CUP_2014/Data/projects.csv'
    path_projectID = '../Data/projectID_train.txt'
    path_res = path_data.split('.')[0] + '_joined_train.' + path_data.split('.')[1]
    join_data_with_projectID(path_data, path_projectID, path_res)
    '''

