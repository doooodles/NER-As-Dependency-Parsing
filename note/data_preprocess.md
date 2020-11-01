# data_preprocess

数据预处理，将提供的data读入并转化为模型可以处理的模式。

#### 数据读入（loader）

将输入的txt文件和目标输出的csv文件读入，根据文件名一一对应，先读入结构较复杂的label为pandas的df，将txt储存的文本读入为str，储存到df的每一行中，df可看作一个数据点，其中每一行代表着每一个实体条目，且每一行都储存有文本信息。

一条信息的内容：dataID，entityID，cat