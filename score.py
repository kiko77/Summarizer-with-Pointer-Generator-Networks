from base64 import decode
import io
from rouge import Rouge 
rouge = Rouge()

file = io.open("result.txt", "r")
test_batch_num = int(file.readline())
print(test_batch_num)
f = [0, 0]
r = [0, 0]
p = [0, 0]
for i in range(test_batch_num):
    decode = file.readline()
    origin = file.readline()
    if i<9:
      print("-----------test "+str(i)+"-----------")
      print("decode:\n", decode)
      print("origin:\n", origin)
    score = rouge.get_scores(hyps=decode, refs=origin)
    for j in range(2):
      f[j] += score[0]['rouge-'+str(j+1)]['f']
      r[j] += score[0]['rouge-'+str(j+1)]['r']
      p[j] += score[0]['rouge-'+str(j+1)]['p']
print("-----------ROUGE-----------")
for k in range(2):
  print("ROUGE-"+str(k+1))
  print("rouge_"+str(k+1)+"_f_score :", f[k]/test_batch_num)
  print("rouge_"+str(k+1)+"_recall :", r[k]/test_batch_num)
  print("rouge_"+str(k+1)+"_precision :", p[k]/test_batch_num, "\n")