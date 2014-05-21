tar xvjf ltrc_yahoo.tar.bz2

python trans_data.py set1.train.txt train train.group

python trans_data.py set1.test.txt test test.group

python trans_data.py set1.valid.txt vali vali.group

../../xgboost ltrc_yahoo.conf

../../xgboost ltrc_yahoo.conf task=pred model_in=0100.model


