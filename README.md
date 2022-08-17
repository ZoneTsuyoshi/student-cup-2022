# student-cup-2022

## score
|branch|CV|LB|rdir|hparams|
|----|----|----|----|----|
|exp02|0.7166850|0.7483100|12-roberta-base2|ep=20, lr=2e-5, do=0.1, FL(2), wd=0.1, sc=None|
|exp9|0.7240025|0.7449114|17-deberta-large1|ep=10, mr=0.1, lr=2e-5, do=0.1, FL(2), wd=0.01, gc=1, sc=Linear(ws=50)|


## MLM pretrained
DeBERTa-v3-base
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|
|----|----|----|----|----|----|----|----|----|
|exp12|1|0.15|2e-5|0.01|8|10|cosine|0.2|