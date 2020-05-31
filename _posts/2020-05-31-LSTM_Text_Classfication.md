### 关于LSTM中最后一个时间步长的输出

来源：[使用LSTM+Pytorch对电影评论进行情感分类](https://zhuanlan.zhihu.com/p/140075236)

```
def get_last_output(self,output,batch_seq_len):
    last_outputs = torch.zeros((output.shape[0],output.shape[2])) #output.shape[0] = batch_size, output.shape[2] = output_size
    for i in range(len(batch_seq_len)): #len(batch_seq_len) == batch_size?
        last_outputs[i] =  output[i][batch_seq_len[i]-1]#index 是长度 -1, 取序列长度seq_len的最后一个索引对应的output值，即为最后一个时间步长的值
    last_outputs = last_outputs.to(output.device)
    return last_outputs
```

### 双向LSTM中将前向和后向的hidden state 拼接起来
```
bidirectional=True：
self.fc = nn.Linear(hidden_dim*2, output_dim)#将前向和后向展平

bidirectional=False：
self.fc = nn.Linear(hidden_dim, output_dim)

hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
```
[解释](https://stackoverflow.com/questions/61012846/how-to-get-final-hidden-state-of-bidirectional-2-layers-gru-in-pytorch)：
The shape[0] of hidden output for bidirectional GRU is 2. You should just concat two hidden output on dim=1:

hid_enc = torch.cat([hid_enc[0,:, :], hid_enc[1,:,:]], dim=1).unsqueeze(0)

As the explanation for usage of -1 and -2 as the index , as you know in python lists, the object in index -1 is the last object
of the list(second object in our tensor list) and index -2 refers to the object before last object(first object in our case). So 
the code you did not understand is equivalent to the code in my answer

扩展：当num_layers > 1, num_layers * num_directions > 2, 此时索引不止两个，猜测0，2，4……属于前向传播，1，3，5……属于后向传播，之后遇到实例之后来更正。
