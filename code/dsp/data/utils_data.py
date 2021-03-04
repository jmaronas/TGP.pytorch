from torchvision.datasets.utils import * #check_integrity 
from torchvision.datasets.folder import * #ImageFolder 
import os
import gzip
import tarfile
import zipfile


def _is_tar(filename):
        return filename.endswith(".tar")


def _is_targz(filename):
        return filename.endswith(".tar.gz") or filename.endswith(".tgz")


def _is_gzip(filename):
        return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
        return filename.endswith(".zip")

def _is_txt(filename):
        return filename.endswith(".txt")

def _is_csv(filename):
        return filename.endswith(".csv")

def extract_archive(from_path, to_path=None, remove_finished=False):                                     
        if to_path is None:
                to_path = os.path.dirname(from_path)                                                     
        
        if _is_tar(from_path):
                with tarfile.open(from_path, 'r') as tar:                                                
                        tar.extractall(path=to_path)                                                     
        
        elif _is_targz(from_path):
                with tarfile.open(from_path, 'r:gz') as tar:                                             
                        tar.extractall(path=to_path)                                                     
        elif _is_gzip(from_path):
                to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])        
                with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:                    
                        out_f.write(zip_f.read())                                                        
        elif _is_zip(from_path):
                with zipfile.ZipFile(from_path, 'r') as z:                                               
                        z.extractall(to_path)                                                            
        elif _is_txt(from_path) or _is_csv(from_path):
            pass
        else:   
                raise ValueError("Extraction of {} not supported".format(from_path))                     
        
        if remove_finished:
                os.remove(from_path)                                                                     
                                                                                                         
def download_and_extract_archive(url, download_root, extract_root=None, filename=None, md5=None, remove_finished=False):  
        download_root = os.path.expanduser(download_root)                                                
        if extract_root is None:                                                                         
                 extract_root = download_root                                                            
        if not filename:                                                                                 
                filename = os.path.basename(url)                                                         
        download_url(url, download_root, filename, md5)                                                  
        archive = os.path.join(download_root, filename)                                                  
        print("Extracting {} to {}".format(archive, extract_root))                                       
        extract_archive(archive, extract_root, remove_finished)  


