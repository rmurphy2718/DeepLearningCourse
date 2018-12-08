CUDA_VISIBLE_DEVICES=3 python main_Bayes.py --num_samples=2 --config_integer=1 --num_folds=10 --num_epochs=20
CUDA_VISIBLE_DEVICES=1 python main_Bayes.py --num_samples=2 --config_integer=2 --num_folds=10 --num_epochs=20

CUDA_VISIBLE_DEVICES=2 python main_Bayes.py --num_samples=2 --config_integer=3 --num_folds=10 --num_epochs=20
CUDA_VISIBLE_DEVICES=3 python main_Bayes.py --num_samples=2 --config_integer=4 --num_folds=10 --num_epochs=20

#
#
CUDA_VISIBLE_DEVICES=0 python main_Bayes.py --num_samples=2 --config_integer=5 --num_folds=10 --num_epochs=20

CUDA_VISIBLE_DEVICES=1 python main_Bayes.py --num_samples=2 --config_integer=6 --num_folds=10 --num_epochs=20
CUDA_VISIBLE_DEVICES=2 python main_Bayes.py --num_samples=2 --config_integer=7 --num_folds=10 --num_epochs=20
CUDA_VISIBLE_DEVICES=3 python main_Bayes.py --num_samples=2 --config_integer=8 --num_folds=10 --num_epochs=20


CUDA_VISIBLE_DEVICES=2 python main_Bayes.py --num_samples=2 --config_integer=9 --num_folds=10 --num_epochs=20
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA_VISIBLE_DEVICES=0 python main_Bayes.py --num_samples=2 --config_integer=1 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'
CUDA_VISIBLE_DEVICES=1 python main_Bayes.py --num_samples=2 --config_integer=2 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'
CUDA_VISIBLE_DEVICES=2 python main_Bayes.py --num_samples=2 --config_integer=3 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'
CUDA_VISIBLE_DEVICES=3 python main_Bayes.py --num_samples=2 --config_integer=4 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'

CUDA_VISIBLE_DEVICES=0 python main_Bayes.py --num_samples=2 --config_integer=5 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'
CUDA_VISIBLE_DEVICES=1 python main_Bayes.py --num_samples=2 --config_integer=6 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'
CUDA_VISIBLE_DEVICES=2 python main_Bayes.py --num_samples=2 --config_integer=7 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'
CUDA_VISIBLE_DEVICES=3 python main_Bayes.py --num_samples=2 --config_integer=8 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'

CUDA_VISIBLE_DEVICES=0 python main_Bayes.py --num_samples=2 --config_integer=9 --num_folds=10 --num_epochs=20 --net_type='2conv3fc'
