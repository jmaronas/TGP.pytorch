echo "UNCOMPRESSING DATA MIGHT TAKE A WHILE... BE PATIENT"

md5sum -c checksum_file.md5

mkdir data_csv
echo "uncompressing the 80s"
# bzip2 -dk compressed/1987.csv.bz2 && mv ./compressed/1987.csv data_csv/
# bzip2 -dk compressed/1988.csv.bz2 && mv ./compressed/1988.csv data_csv/
# bzip2 -dk compressed/1989.csv.bz2 && mv ./compressed/1989.csv data_csv/
echo "uncompressing the 90s"
# bzip2 -dk compressed/1990.csv.bz2 && mv ./compressed/1990.csv data_csv/
# bzip2 -dk compressed/1991.csv.bz2 && mv ./compressed/1991.csv data_csv/
# bzip2 -dk compressed/1992.csv.bz2 && mv ./compressed/1992.csv data_csv/
# bzip2 -dk compressed/1993.csv.bz2 && mv ./compressed/1993.csv data_csv/
# bzip2 -dk compressed/1994.csv.bz2 && mv ./compressed/1994.csv data_csv/
# bzip2 -dk compressed/1995.csv.bz2 && mv ./compressed/1995.csv data_csv/
# bzip2 -dk compressed/1996.csv.bz2 && mv ./compressed/1996.csv data_csv/
# bzip2 -dk compressed/1997.csv.bz2 && mv ./compressed/1997.csv data_csv/
# bzip2 -dk compressed/1998.csv.bz2 && mv ./compressed/1998.csv data_csv/
# bzip2 -dk compressed/1999.csv.bz2 && mv ./compressed/1999.csv data_csv/
echo "uncompressing 2000s"
# bzip2 -dk compressed/2000.csv.bz2 && mv ./compressed/2000.csv data_csv/
# bzip2 -dk compressed/2001.csv.bz2 && mv ./compressed/2001.csv data_csv/
# bzip2 -dk compressed/2002.csv.bz2 && mv ./compressed/2002.csv data_csv/
# bzip2 -dk compressed/2003.csv.bz2 && mv ./compressed/2003.csv data_csv/
# bzip2 -dk compressed/2004.csv.bz2 && mv ./compressed/2004.csv data_csv/
# bzip2 -dk compressed/2005.csv.bz2 && mv ./compressed/2005.csv data_csv/
# bzip2 -dk compressed/2006.csv.bz2 && mv ./compressed/2006.csv data_csv/
# bzip2 -dk compressed/2007.csv.bz2 && mv ./compressed/2007.csv data_csv/
bzip2 -dk compressed/2008.csv.bz2 && mv ./compressed/2008.csv data_csv/

