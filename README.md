# student-cup-2022

## score
|branch|CV|LB|rdir|hparams|
|----|----|----|----|----|
|exp02|0.7166850|0.7483100|12-roberta-base2|ep=20, lr=2e-5, do=0.1, FL(2), wd=0.1, sc=None|
|exp9|0.7240025|0.7449114|17-deberta-large1|ep=10, mr=0.1, lr=2e-5, do=0.1, FL(2), wd=0.01, gc=1, sc=Linear(ws=50)|
|exp10|0.695|0.7143739|17-deberta-base1|ep=10, mr=0.1, lr=2e-5, do=0.1, FL(2), wd=0.01, gc=1, sc=Linear(ws=50), at=AWP(eps=1e-2,lr=1e-4)|


## MLM pretrained
DeBERTa-v3-base
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp12|1|0.15|2e-5|0.01|8|10|cosine|0.2|8x4=32|
|exp13|2|0.1|2e-5|0.01|8|10|cosine|0.2|8x4=32|
|exp13|3|0.1|2e-5|0.01|8|20|cosine|0.2|16x4=64|
|exp13|4|0.15|2e-5|0.01|8|10|cosine|0.2|8x4=32|
|exp13|5|0.15|2e-5|0.01|8|20|cosine|0.2|16x4=64|
|exp13|6|0.1|2e-5|0.01|8|10|cosine|0.2|8x4+=|
|exp13|7|0.15|2e-5|0.01|8|10|cosine|0.2|8x4+=|

DeBERTa-v3-large
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp14|1|0.1|2e-5|0.01|4|10|cosine|0.2||
|exp14|2|0.15|2e-5|0.01|4|10|cosine|0.2||

DeBERTa-base
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp13|1|0.1|2e-5|0.01|8|10|cosine|0.2|12.25x4=49|
|exp13|2|0.1|2e-5|0.01|8|20|cosine|0.2|23.75x4=95|
|exp13|3|0.15|2e-5|0.01|8|10|cosine|0.2|12.25x4=49|
|exp13|4|0.15|2e-5|0.01|8|20|cosine|0.2|23.75x4=95|
|exp13|5|0.1|2e-5|0.01|8|10|cosine|0.2|12.25x4+=|
|exp13|6|0.15|2e-5|0.01|8|10|cosine|0.2|12.25x4+=|

DeBERTa-large
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp14|1|0.1|2e-5|0.01|4|10|cosine|0.2||
|exp14|2|0.15|2e-5|0.01|4|10|cosine|0.2||

RoBERTa-base
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp13|1|0.1|2e-5|0.1|8|10|cosine|0.2|6.5x4=26|
|exp13|2|0.1|2e-5|0.1|8|20|cosine|0.2|12.75x4=51|
|exp13|3|0.15|2e-5|0.1|8|10|cosine|0.2|6.5x4=26|
|exp13|4|0.15|2e-5|0.1|8|20|cosine|0.2|12.75x4=51|
|exp14|5|0.1|2e-5|0.1|8|10|cosine|0.2|6.5x4+7=33|
|exp14|6|0.15|2e-5|0.1|8|10|cosine|0.2|6.5x4+7=33|

RoBERTa-large
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp13|1|0.1|2e-5|0.1|4|10|cosine|0.2|17.25x4=69|
|exp13|2|0.1|2e-5|0.1|4|20|cosine|0.2|34.25x4=137|
|exp13|3|0.15|2e-5|0.1|4|10|cosine|0.2|17.25x4=69|
|exp13|4|0.15|2e-5|0.1|4|20|cosine|0.2|34.25x4=137|
|exp14|5|0.1|2e-5|0.1|4|10|cosine|0.2|17.25x4+=|
|exp14|6|0.15|2e-5|0.1|4|10|cosine|0.2|17.25x4+=|