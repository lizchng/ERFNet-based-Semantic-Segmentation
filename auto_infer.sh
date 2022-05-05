cd /media/personal_data/lizc/1/erfnet_pytorch/eval/
export CUDA_VISIBLE_DEVICES=1
python -W ignore eval_zhoushan.py --loadDir ../save/group0003/ --datadir ../dataset/unlabeled/group0003
python -W ignore eval_zhoushan.py --loadDir ../save/group0004/ --datadir ../dataset/unlabeled/group0004
python -W ignore eval_zhoushan.py --loadDir ../save/group0005/ --datadir ../dataset/unlabeled/group0005
python -W ignore eval_zhoushan.py --loadDir ../save/group0010/ --datadir ../dataset/unlabeled/group0010
python -W ignore eval_zhoushan.py --loadDir ../save/group0015/ --datadir ../dataset/unlabeled/group0015
python -W ignore eval_zhoushan.py --loadDir ../save/group0018/ --datadir ../dataset/unlabeled/group0018
python -W ignore eval_zhoushan.py --loadDir ../save/group0021/ --datadir ../dataset/unlabeled/group0021
python -W ignore eval_zhoushan.py --loadDir ../save/group0026/ --datadir ../dataset/unlabeled/group0026
python -W ignore eval_zhoushan.py --loadDir ../save/group0028/ --datadir ../dataset/unlabeled/group0028
