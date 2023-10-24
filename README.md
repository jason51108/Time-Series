# 时间序列仓库
本代码仓库是一个面向深度学习研究人员的开源库，特别是深度时间序列分析。

提供了一个整洁的代码库来评估先进的深度时间序列模型或开发自己的模型，它涵盖了五个主流任务:**long- and short-term forecasting, imputation, anomaly detection, and classification.**

## 时间序列分析排行榜

截至2023年2月，五种不同任务的前三名模型是:

| Model<br>Ranking | Long-term<br>Forecasting                                     | Short-term<br>Forecasting                                    | Imputation                                                   | Anomaly<br>Detection                                         | Classification                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 🥇 1st            | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)       |
| 🥈 2nd            | [DLinear](https://github.com/cure-lab/LTSF-Linear)           | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd            | [Non-stationary<br>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer)           | [Autoformer](https://github.com/thuml/Autoformer)            | [Informer](https://github.com/zhouhaoyi/Informer2020)        | [Autoformer](https://github.com/thuml/Autoformer)  |

**比较这个模型的排行榜.** ☑ 意味着它们的代码已经包含在这个repositories中

  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py)
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py)
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py)
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py)
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py)
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py)
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py)
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py)
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py)
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)

 **Newly added baselines.** 将在综合评估后将其添加到排行榜中。

------

  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py)

  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py)
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py)

## 使用方法

1. 
   安装Python 3.8。为方便起见，执行以下命令。

```python
pip install -r requirements.txt
```

2. 准备数据. 你可以从以下渠道获取数据 [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/84fbc752d0e94980a610/) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). 然后将下载的数据放在文件夹下 `./data_provider/dataset`. 下面是支持的数据集的摘要。当需要自定义的数据集时，需要将

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. 在`./scripts/`文件夹下提供了所有基准测试的实验脚本。您可以将实验结果复制为以下示例:

   ```python
   # long-term forecast
   bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
   # short-term forecast
   bash ./scripts/short_term_forecast/TimesNet_M4.sh
   # imputation
   bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
   # anomaly detection
   bash ./scripts/anomaly_detection/PSM/TimesNet.sh
   # classification
   bash ./scripts/classification/TimesNet.sh
   ```

   若为Window操作系统，您可以按以下代码执行示例：

   ```python
   cd C:\Users\user\Desktop\Time-Series
   python -u run.py --task_name long_term_forecast  --model informer --data ETTh1
   ```

4. 开发你自己的模型。

- 将模型文件添加到`./models`文件夹中。你可以按照 `./models/Transformer.py`.
- 将新添加的模型包含在 `./exp/exp_basic.py`的 `Exp_Basic.model_dict`中
- 在文件夹下创建相应的脚本 `./scripts`.

## 联络
如有任何疑问或建议，欢迎联络，或者可以提相关的Issues，本人在看见后会尽可能解答

- 原作者：Haixu Wu (wuhx23@mails.tsinghua.edu.cn)
  - 整理各时间序列相关的源代码
- 本人: Yinghang Chen ([brainiaccc@foxmail.com]())
  - 增加代码中文注释；增加了可接受DataFrame的自定义的Dataloader类；修改源代码中可能存在的数据结构冗余等问题

## 关于

国家重点研发计划项目(2021YFB1715200)资助。

这个库是基于以下代码库构建的:

- Forecasting: https://github.com/thuml/Autoformer

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://github.com/thuml/Flowformer

所有实验数据集都是公开的，可以下链接获取:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://www.timeseriesclassification.com/
