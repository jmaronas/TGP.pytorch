echo "PROCESSING 2008 DATA.... FILTERING FEATURES"
cd data_csv
wget http://stat-computing.org/dataexpo/2009/plane-data.csv

cat plane-data.csv |  awk -F , '{print $1, $9}' > /tmp/plane-data-processed # get tail identifier and year of construction
cat 2008.csv |  awk -F, 'BEGIN{}{ if (NR != 1 && $22 == 0 && $24 == 0 ) {print $11}}END{}' > /tmp/tail-id # the the corresponding tail id
cat 2008.csv |  awk -F, 'BEGIN{}{ if (NR != 1 && $22 == 0 && $24 == 0 ) {print $2","$3","$4","$5","$7","$14","$19","$15}}END{}' > /tmp/raw-features # get the raw features by filtering cancelled flights and also delivered (cols 22 and 24)
cd ..

python process_tail_vs_year.py > /tmp/plane-year

paste -d "," /tmp/plane-year /tmp/raw-features | awk -F, 'BEGIN{}{ if ( $1 != "NA" ) {print $1","$2","$3","$4","$5","$6","$7","$8","$9}}END{}' > ./data_csv/2008_processed.csv #filter plane years that were not available


