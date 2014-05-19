python trans_data.py train.txt mq2007-list.train mq2007-list.train.group

python trans_data.py test.txt mq2007-list.test mq2007-list.test.group

python trans_data.py vali.txt mq2007-list.vali mq2007-list.vali.group

../../xgboost mq2007-list.conf

../../xgboost mq2007-list.conf task=pred model_in=0004.model


