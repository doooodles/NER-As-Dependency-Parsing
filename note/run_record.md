# run model

在/work_NER/NewNER下调用train.py，参数需要在脚本中设定。训练过程不会对XLnet预训练模型进行微调训练，每一个epoch中，选择3/4的数据用于训练，训练结束后调用eva函数进行验证，并对所有的epoch保留最小的dev_loss和最优模型的模型参数。

```bash
CUDA_VISIBLE_DEVICES='1' CUDA_LAUNCH_BLOCKING=1  python -u train.py > ../log/test.log 2>&1 &

```

训练过程的交叉熵损失默认为mean模式，但平均的结果会导致loss数值较小，可考虑使用sum模式，但同时梯度也会变大。

## nni调参

依然在/work_NER/NewNER，采用nni框架，调用train_parser.py来进行自动调参，train_parser.py会自动接受命令行传递的参数设置，对不同的参数设置进行固定轮次的训练。

```bash
# 启动nni调参，更改gpu设置时需要对config文件做修改
CUDA_VISIBLE_DEVICES='5' nnictl create --config NewNER/config.yml --port 8890

# 如果发生问题及时停止
nnictl stop
```

第一轮次较为粗糙的调参的搜索空间设置：

```json
{
    "epoch": {"_type":"choice", "_value": [64]},
    "batch_size": {"_type":"choice", "_value": [8,12]},
    "d_in":{"_type":"choice","_value":[768]},
    "d_hid":{"_type":"choice","_value":[384,512,768,1024]},
    "lr":{"_type":"choice","_value":[0.0001,0.005,0.001]},
    "dropout":{"_type":"choice","_value":[0.3,0.4,0.2,0.5]},
    "n_laye