dir='./downloaded_data/sic97data_01/'
mv $dir'sic_full.dat' $dir'sic_full.dat.backup'
cat $dir'sic_full.dat.backup' | awk 'BEGIN{}{ if ( NR > 5 && NR == 6 ){print "id,x,y,rainfall"}; if( NR > 6) {print $0} }' > $dir'sic_full.dat'

python setup.py
mv ./data/* ..

