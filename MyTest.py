sent=["5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","1","2","3","4","。","5","5","5","。","5","5","5","5","5","5","5","5","5","5"]
biaodian=["。", ",", "，", "!", "！", "?", "？", "、", "；"]

start=0
sent_all=[]
flag=0
for i in range(len(sent)):
    if sent[i] =="。":
        flag = i
    elif i >10:
        if flag >0:
            break
        else:
            flag=10
print(flag)

print(sent_all)
split_sent = []
sents=[]
# @why 超过max_len的长句子按符号列表切片
start = 0
print("done")