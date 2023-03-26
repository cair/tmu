if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi
#conda activate tmu
optuna-dashboard mysql+pymysql://$MYSQL_USER:$MYSQL_PASS@localhost/$MYSQL_DB --host=0.0.0.0
