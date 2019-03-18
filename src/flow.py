from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'Jayson',
    'depends_on_past': False,
    'start_date': datetime(2019, 3, 17),
    'email': ['790930856@qq.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG('asp-mtl', default_args=default_args, schedule_interval=timedelta(days=1))

batch_sizes = [16, 32, 64]
hidden_sizes = [64, 128, 256]
vocab_sizes = [8, 16, 32, 64]
num_epochs = 20
lrn_rates = [0.01, 0.005, 0.001]

bash_command = "print cd /mnt/d/data/fudan_mtl_reviews/src && python main.py --batch_size={} --hidden_size={} --vocab_size={} --symbol_dropout=0.1 --adv_weight=0.05 " \
               "--l2_coef=0 --keep_prob=0.7 --num_epochs={} --model=lstm --use_attention " \
               "--diff_weight=50000 --adv --subword --attention_diff --lrn_rate={} --logdir={}"

o = None
task_id = 0

#
# for batch_size in batch_sizes:
#     for hidden_size in hidden_sizes:
#         for vocab_size in vocab_sizes:
#             for lrn_rate in lrn_rates:
#                 logdir = "saved_models/{}-{}-{}-{}-{}/".format(batch_size, hidden_size, vocab_size, num_epochs, lrn_rate)
#                 bc = bash_command.format(batch_size,
#                                          hidden_size,
#                                          vocab_size,
#                                          num_epochs,
#                                          lrn_rate,
#                                          logdir)
#                 t = BashOperator(
#                     task_id='asp-mtl-{}'.format(task_id),
#                     bash_command=bc,
#                     dag=dag)
#
#                 if o:
#                     o >> t
#                 o = t
#
#                 task_id = task_id + 1


batch_size = 16
hidden_size = 64
vocab_size = 8
lrn_rate = 0.01

logdir = "saved_models/{}-{}-{}-{}-{}/".format(batch_size, hidden_size, vocab_size, num_epochs, lrn_rate)
bc = bash_command.format(batch_size,
                         hidden_size,
                         vocab_size,
                         num_epochs,
                         lrn_rate,
                         logdir)
t = BashOperator(
    task_id='asp-mtl-{}'.format(task_id),
    bash_command="print abc",
    dag=dag)

# if o:
#     o >> t
# o = t
