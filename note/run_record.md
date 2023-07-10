# run model

在/work_NER/NewNER下调用train.py，参数需要在脚本中设定。训练过程不会对XLnet预训练模型进行微调训练，每一个epoch中，选择3/4的数据用于训练，训练结束后调用eva函数进行验证，并对所有的epoch保留最小的dev_loss和最优模型的模型参数。

```bash
CUDA_VI