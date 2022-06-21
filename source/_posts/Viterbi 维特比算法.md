---
dated: 2022-05-27 11:03
updated: 2022-06-21 16:00
title: Viterbi 维特比算法

categories:
  - Python

tags:
  - 算法
---

# Viterbi 维特比算法

相关企业：

动态规划中蛮经典的算法，尤其是在 HMM&HSMM 中，解决算法都是基于维特比实现的。

对于维特比算法的简单理解，就是每一步都遍历当前各种状态下当前发生概率最大的事件，根据最后时刻发生概率最大的事件，逐一回溯，找出其发生的路径。

## 伪代码

[维基百科](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95) 的说明比较好理解，伪代码基于该内容。

想象一个乡村诊所。村民有着非常理想化的特性，要么健康要么发烧。他们只有问诊所的医生的才能知道是否发烧。 聪明的医生通过询问病人的感觉诊断他们是否发烧。村民只回答他们感觉正常、头晕或冷。

假设一个病人每天来到诊所并告诉医生他的感觉。医生相信病人的健康状况如同一个离散马尔可夫链。病人的状态有两种“健康”和“发烧”，但医生不能直接观察到，这意味着状态对他是“隐含”的。每天病人会告诉医生自己有以下几种由他的健康状态决定的感觉的一种：正常、冷或头晕。这些是观察结果。 整个系统为一个隐马尔可夫模型 (HMM)。

医生知道村民的总体健康状况，还知道发烧和没发烧的病人通常会抱怨什么症状。假设病人为 A，下方是相关参数。

```python
# A 可能存在的状态
states = ('Healthy', 'Fever')

# 医生能够观察的现象
observations = ('normal', 'cold', 'dizzy')

# A 最开始状态的概率
start_probability = {'Healthy': 0.6, 'Fever': 0.4}

# 从健康转移感冒，或者相反的概率
transition_probability = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6},
   }

# 发射概率，也就是不同状态下，医生能观察到现象的概率
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }

# A 近 7 天观察的现象记录
record = ['normal', 'cold', 'cold', 'dizzy', 'normal', 'cold', 'cold']
```

医生去判断 A 近七天的健康状态，就是维特比算法要解决的问题。

第一步：计算第一天各状态的概率

```python
# 最开始是健康的，且第一天也是健康的概率
healthy_healthy_prob = start_probability['Healthy'] * transition_probability['Healthy']['Healthy'] * emission_probability['Healthy']['normal']
# 最开始是感冒的，但第一天也是健康的概率
fever_healthy_prob = start_probability['Fever'] * transition_probability['Fever']['Healthy'] * emission_probability['Healthy']['normal']
# 最开始是感冒的，且第一天也是感冒的概率
fever_fever_prob = start_probability['Fever'] * transition_probability['Fever']['Fever'] * emission_probability['Fever']['normal']
# 最开始是健康的，但第一天也是感冒的概率
healthy_fever_prob = start_probability['Healthy'] * transition_probability['Healthy']['Fever'] * emission_probability['Fever']['normal']
```

第二步，参考第一步的过程依次计算到最后一天各状态的概率
第三步，从最后一天开始回溯： - 如果 healthy_healthy_prob 的概率最大，那么最后一天就是健康的，且前一天也是健康的； - 如果 fever_healthy_prob 的概率最大，那么最后一天是健康的，但前一天是感冒的； - 如果 fever_fever_prob 的概率最大，那么最后一天就是感冒的，且前一天也是感冒的； - 如果 healthy_fever_prob 的概率最大，那么最后一天就是感冒的，但前一天也是健康的；
第四步：参考第三步，不断回溯，得出病人的健康状态。

## 代码示例

代码应用于 [[hw2-2021-phoneme 识别]] 中，对模型的输出进行后处理，找出发生概率最大的事件。以下代码充分利用了 numpy 矩阵计算的特点，计算效率特别快。

```python
# 状态迁移的概率，如 A 变成 B 概率
trans_table_norm = trans_table / np.sum(trans_table, axis=1, keepdims=True)
trans_table_norm = prob + 1e-17
# 使用log函数，是为了在后续大量计算中使用加分，而非乘法，提高效率、降低内存
trans_table_norm = np.log(trans_table_norm)

m = nn.Softmax(1)
raw_output = torch.tensor(output)
test_ln_softmax = m(raw_output)
test_ln_softmax = np.array(test_ln_softmax)
test_ln_softmax = test_ln_softmax + 1e-17
# 发射概率，既可能是 A 的概率
test_ln_softmax = np.log(test_ln_softmax)

# 记录每个观察点的概率分布
tracking = np.zeros((451552, 39))
# 开始状态的概率分布
last_state = test_ln_softmax[0]
for i in range(1, len(test_ln_softmax)):
    # 计算当前节点每种状态的发生概率
    prob = last_state.reshape(39, 1) + trans_table_norm +  test_ln_softmax[i]
    # 找出当前节点的各种状态下发生的最大概率
    current_state = np.max(prob, axis=0)
    # 找出当前节点的各种状态下发生的最大概率时，前一节点的状态
    tracking[i] = np.argmax(prob, axis=0)
    last_state = current_state

# 从末尾开始回溯最优节点
pred_ls = [np.argmax(raw_output[-1])]
for i in range(0, 451551):
    back = tracking[451552-i-1][int(pred_ls[-1])]
    pred_ls.append(int(back))
t_predict = pred_ls[::-1]
```
