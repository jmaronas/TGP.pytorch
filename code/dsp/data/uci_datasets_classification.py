import os
import numpy
import pandas as pd
import warnings
from .utils_data import download_and_extract_archive
from .uci_datasets import UCI_data

class Avila(UCI_data):

    def __init__(self, use_validation = None, split_from_disk = True) -> None :
        # https://archive.ics.uci.edu/ml/datasets/Avila
        self.url        = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip'
        self.name       = 'avila'
        self.tr_md5sum  = 'b78ee4f810c6de0e0c933ef36345f53b'
        self.te_md5sum  = 'b0eb4691d779086bcd73ae4951b891ee' 
        self.tr_name    = 'avila-tr.txt' 
        self.te_name    = 'avila-ts.txt'
        self.n_classes  = 12 

        self.sep        = ',' 
        self.header     = None

        self.use_validation = use_validation

        super().__init__('classification',normalize_y = False, download = True, split_from_disk = split_from_disk)

        self.cast_outputs()

    def __load__(self,split_from_disk):

        # We cant use the common functionality due to how avila is given 
        fpath = self.root
        if self.download:
            if self._check_integrity(fpath,self.tr_name,self.tr_md5sum) and self._check_integrity(fpath,self.te_name,self.te_md5sum):
                print('Files already downloaded and verified')
            else:
                download_and_extract_archive(self.url, self.directory, remove_finished = True)
        if not self._check_integrity(fpath,self.tr_name,self.tr_md5sum):
            raise Exception("Files corrupted. Set download=True") 

        if not self._check_integrity(fpath,self.te_name,self.te_md5sum):
            raise Exception("Files corrupted. Set download=True") 


        # load data 
        fpath_tr = os.path.join(self.root,self.tr_name)
        data_tr = self.load_csv_data(fpath_tr, shuffle = False, seed = None, sep = self.sep, header = self.header, return_pandas = True) # train and test is already given for this dataset


        fpath_te = os.path.join(self.root,self.te_name)
        data_te = self.load_csv_data(fpath_te, shuffle = False, seed = None, sep = self.sep, header = self.header, return_pandas = True) # train and test is already given for this dataset


        data_tr[10] = pd.Categorical(data_tr[10])
        data_tr[10] = data_tr[10].cat.codes 

        data_te[10] = pd.Categorical(data_te[10])
        data_te[10] = data_te[10].cat.codes 

        tr_labels = data_tr[10].unique()
        te_labels = data_te[10].unique()

        tr_labels.sort()
        te_labels.sort()

        assert (tr_labels == te_labels).all(), "Different number of numerical categorical labels in training and test"

        X_tr = data_tr.to_numpy()
        X_te = data_te.to_numpy()

        X_tr,Y_tr = X_tr[:,0:10],X_tr[:,10:].astype('int')
        X_te,Y_te = X_te[:,0:10],X_te[:,10:].astype('int')

        return X_tr, Y_tr, X_te, Y_te



class Banknote(UCI_data):

    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        # https://archive.ics.uci.edu/ml/datasets/Avila
        self.url        = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
        self.name       = 'data_banknote_authentication.txt'
        self.md5sum     = '3f64e2b50525a2f36dcd947eaf7bac8a' 
        self.n_classes  = 2

        self.shuffle         = True
        self.seed            = seed
        self.sep             = ',' 
        self.header          = None
        self.index           = -1

        self.use_validation = use_validation

        super().__init__('classification',normalize_y = False, download = True, remove_finished = False, split_from_disk = split_from_disk)

        self.cast_outputs()

class Movement(UCI_data):
    def __init__(self,seed,use_validation = None, split_from_disk = True):
        # https://archive.ics.uci.edu/ml/datasets/Indoor+User+Movement+Prediction+from+RSS+data 
        self.dir_download = '/tmp/' 
        self.url          = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00348/MovementAAL.zip'

        self.md5sum    = '83a84156b5693029f13a9a3b02b176b2'
        self.name      = 'movement.csv' 
        self.n_classes = 2

        self.shuffle   = True 
        self.seed      = seed
        self.sep       = ',' 
        self.header    = None
        self.index     = -1

        self.use_validation = use_validation

        super().__init__('classification',normalize_y = False, download = True, split_from_disk = split_from_disk)

        self.cast_outputs()

    def _preprocess(self):
        fpath        = os.path.join(self.dir_download,'dataset')
        fpath_labels = os.path.join(fpath,'MovementAAL_target.csv')

        # Read labels first. Each possition corresponds to each identifier below
        labels = self.load_csv_data(fpath_labels, shuffle = False, seed = None, sep = self.sep, header = 0, return_pandas = False)[:,1]

        DATA   = numpy.empty((0,5), dtype = 'float64')

        file2index = os.listdir(fpath)
        file2index.sort()

        for _file in file2index:
            if _file == 'MovementAAL_target.csv':
                continue
            else:
                fpath_data = os.path.join(fpath,_file)
                id_ = int(_file.split(".csv")[0].split("_")[-1])-1 # index starts at zero

                data_ = self.load_csv_data(fpath_data, shuffle = False, seed = None, sep = self.sep, header = 0, return_pandas = False)

                lab = numpy.ones((data_.shape[0],1))
                lab = lab*labels[id_] if labels[id_] == 1 else lab*0.0

                data_ = numpy.hstack((data_,lab))

                DATA   = numpy.vstack((DATA,data_))

        numpy.random.seed(0)
        for i in range(20):
            numpy.random.shuffle(DATA)

        fpath_out = os.path.join(self.directory,self.name)
        numpy.savetxt(fpath_out, DATA, delimiter=",")

    def __load__(self,split_from_disk):
        fpath = self.directory
        if self.download:
            if self._check_integrity(fpath,self.name,self.md5sum):
                print('Files already downloaded and verified')
            else:
                download_and_extract_archive(self.url, self.dir_download, remove_finished = True)
                self._preprocess()

        if not self._check_integrity(fpath,self.name,self.md5sum):
            raise Exception("Files corrupted. Set download=True") 

        return super().__load__(split_from_disk)


class Activity(UCI_data):
    def __init__(self,seed,use_validation = None, split_from_disk = True):
        # https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM%29 
        self.url          = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00366/AReM.zip'
        self.dir_download = '/tmp/activity/'

        self.name      = 'activity.csv'
        self.md5sum   = 'b94c98c59e1791891cf3bb9c277fd8a4'
        self.n_classes = 7

        self.shuffle   = True 
        self.seed      = seed
        self.sep       = ',' 
        self.header    = None
        self.index     = -1

        self.use_validation = use_validation

        self.solve_parsing_error_type1 = [self.dir_download+'cycling/dataset14.csv',self.dir_download+'cycling/dataset9.csv']
        self.solve_parsing_error_type2 = [self.dir_download+'bending2/dataset4.csv']

        super().__init__('classification',normalize_y = False, download = True, split_from_disk = split_from_disk)

        self.cast_outputs()

    def _solve_parsing_error(self,file_):

        number_lines = 0
        # count number of lines
        for idx,line in enumerate(open(file_,'r')):
            number_lines += 1

        new_file_content = ""

        if file_ in self.solve_parsing_error_type1:
            for idx,line in enumerate(open(file_,'r')):
                if idx+1 == number_lines:
                    if len(line.split(',')) == 8:
                        line_split = line.split(',')[0:-1]
                        line       = ','.join(line_split)
                new_file_content += line
        else:
            for idx,line in enumerate(open(file_,'r')):

                append = "" if idx+1 == number_lines else "\n"
                if idx > 4:
                    if len(line.split(",")) < 5: # just do it when "," has not been incorporated
                        line = ",".join(line.split(" ")[0:-1])

                new_file_content += line + append

        writing_file = open(file_, "w")
        writing_file.write(new_file_content)
        writing_file.close()

    def _preprocess(self):

        classes = ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']
        DATA    = numpy.empty((0,7))
        for lab,c in enumerate(classes):
            fpath        = os.path.join(self.dir_download,c)

            file2index = os.listdir(fpath)
            file2index.sort() # sort otherwise the md5sum wont be consistent across different filesystems

            for _file in file2index:
                fpath_data = os.path.join(fpath,_file)

                if fpath_data in self.solve_parsing_error_type1 or fpath_data in self.solve_parsing_error_type2:
                    self._solve_parsing_error(fpath_data)

                data_ = self.load_csv_data(fpath_data, shuffle = False, seed = None, sep = self.sep, header = 4, return_pandas = False)[:,1:]


                if fpath_data == '/tmp/activity/sitting/dataset8.csv':
                    warnings.warn('This file {} has one few datapoint than what the repository claims'.format(fpath_data),RuntimeWarning)
                labels = numpy.ones((data_.shape[0],1)) * lab

                #print(data_.shape)
                #print(DATA.shape)
                data_ = numpy.hstack((data_,labels))
                DATA   = numpy.vstack((DATA,data_))
        
        numpy.random.seed(0)
        for i in range(20):
            numpy.random.shuffle(DATA)

        fpath_out = os.path.join(self.directory,self.name)
        numpy.savetxt(fpath_out, DATA, delimiter=",")

    def __load__(self,split_from_disk):
        
        # We cant use the common functionality due to how Activity is given
        fpath = self.directory
        if self.download:
            if self._check_integrity(fpath,self.name,self.md5sum):
                print('Files already downloaded and verified')
            else:
                download_and_extract_archive(self.url, self.dir_download, remove_finished = True)
                self._preprocess()

        if not self._check_integrity(fpath,self.name,self.md5sum):
            raise Exception("Files corrupted. Set download=True") 

        return super().__load__(split_from_disk)


class Heart(UCI_data):

    def __init__(self,seed,use_validation = None, split_from_disk = True):
        # https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records 
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'

        self.name            = 'heart_failure_clinical_records_dataset.csv'
        self.md5sum          = '690e98e799498994da318807f5c5f476'
        self.n_classes       = 2
        self.are_categorical = numpy.array([1,3,5,9,10]) # to avoid normalizing these ones

        self.shuffle    = True
        self.seed       = seed
        self.sep        = ',' 
        self.header     = 0
        self.index      = -1

        self.use_validation = use_validation

        super().__init__('classification',normalize_y = False, download = True, remove_finished = False, split_from_disk = split_from_disk)

        self.cast_outputs()
