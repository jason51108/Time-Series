import argparse
import time
import os
import torch

# 时间序列的五大任务
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

if __name__ == '__main__':
    # 在Vscode这个IDE中偶发性地出现相对路径不正确的错误
    # os.chdir('your path')

    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='时间序列代码库')

    # 基础配置
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='训练状态:0代表无需训练; 1代表需要训练, options: [0,1]') 
    parser.add_argument('--model_id', type=str, default='test', help='模型id:train代表测试集;vail代表验证集;test代表测试集, options: [train,vail,test]')
    parser.add_argument('--model', type=str, default='Informer', required=True, help='模型名字, options: [Autoformer, Transformer, TimesNet, Informer.....]')

    # data loader配置
    parser.add_argument('--data', type=str, default='ETTh1', required=True, help='数据集类型,如果为自定义数据集则输入custom')
    parser.add_argument('--root_path', type=str, default='./data_provider/dataset/', help='数据文件根目录')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件')
    parser.add_argument('--features', type=str, default='MS',
                        help='预测任务, options:[M, S, MS]; M:多对多, S:单对单, MS:多对单')
    parser.add_argument('--target', type=str, default='OT', help='S或MS中DataFrame的目标列,若为M则可随便填写')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间戳编码的频率, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], \
                              同样的你可以采取15min或3h这种方式表示频率。注意, 如果为d的时候time类型会转为[a,b,c]的形式, 这在采用Embedding的时候会出错')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点位置')

    # forecasting task(预测任务配置这里)
    parser.add_argument('--seq_len', type=int, default=96, help='encoder输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='Encoder和Decoder共有部分的长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测长度')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4(只针对季节性数据集)')
    parser.add_argument('--inverse', action='store_true', help='逆归一化输出结果', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task(异常检测任务)
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # 模型定义
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder输入维度')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder输入维度')
    parser.add_argument('--c_out', type=int, default=1, help='预测数据输出维度')
    parser.add_argument('--d_model', type=int, default=512, help='升维到多少维度')
    parser.add_argument('--n_heads', type=int, default=8, help='多头注意力的头数')
    parser.add_argument('--e_layers', type=int, default=2, help='encoder层数')
    parser.add_argument('--d_layers', type=int, default=1, help='decoder层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout比率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数选择')
    parser.add_argument('--output_attention', action='store_true', help='是否在encoder中输出注意力')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader的workers数量')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='训练数据的batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.000001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # cuda
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 打印总的参数
    print('Args in experiment:')
    print(args)

    # 根据任务不同选择做不同的选择
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    # 如果需要训练，则先进行训练，然后test;如果不需要训练，则从保存模型的地址加载模型参数model.load_state_dict
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
    end_ = time.time()

