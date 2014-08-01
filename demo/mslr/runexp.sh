python trans_data.py train.txt train train.group

python trans_data.py test.txt test test.group

python trans_data.py vali.txt vali vali.group

../../xgboost rank.conf

../../xgboost rank.conf task=pred model_in=0004.model


