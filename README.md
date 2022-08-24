# student-cup-2022

## score
|branch|CV|LB|rdir|hparams|
|----|----|----|----|----|
|exp02|0.717|0.748|12-roberta-base2|ep=20, lr=2e-5, do=0.1, FL(2), wd=0.1, sc=None|
|exp9|0.724|0.745|17-deberta-v3-large1|ep=10, mr=0.1, lr=2e-5, do=0.1, FL(2), wd=0.01, gc=1, sc=Linear(ws=50)|
|exp10|0.695|0.714|17-deberta-base1|ep=10, mr=0.1, lr=2e-5, do=0.1, FL(2), wd=0.01, gc=1, sc=Linear(ws=50), at=AWP(eps=1e-2,lr=1e-4)|
|exp15|0.717|0.718|19_gs1/mnroberta-basebs16wd0.1mi5umFalselFLg2aawpal1.0g0|ep=10, mr=0,1, lr=2e-5, do=0.1, sc=Linear(ws=100)|
|exp15|0.715|0.710|19_gs1/mnxlnet-base-casedbs16wd0.01mi1umFalselFLg2aawpal1.0g0|ep=10, mr=0.1, lr=2e-5, do=0.1, sc=Linear(ws=100)|
|exp16|0.692|0.729|22_gs1/mnmicrosoft/deberta-v3-largebs8wd0.01e10aNoneal1.0g2|mr=0.1, lr=2e-5, do=0.1, sc=Linear(ws=100)|
|exp16|0.716|0.749|22_gs1/mnroberta-largebs8wd0.1e10mi5umTruel2e-05g0|mr=0.1, do=0.1, sc=Linear(ws=100)|
|exp16|0.722|0.726|23_gs1/mnmicrosoft/deberta-v3-largebs8wd0.01e10mi1mr0.1afgmal0.1g0|do=0.1, sc=Linear(ws=100)|

## MLM pretrained
DeBERTa-v3-base
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp12|1|0.15|2e-5|0.01|8|10|cosine|0.2|8x4=32|
|exp13|2|0.1|2e-5|0.01|8|10|cosine|0.2|8x4=32|
|exp13|3|0.1|2e-5|0.01|8|20|cosine|0.2|16x4=64|
|exp13|4|0.15|2e-5|0.01|8|10|cosine|0.2|8x4=32|
|exp13|5|0.15|2e-5|0.01|8|20|cosine|0.2|16x4=64|
|exp13|6|0.1|2e-5|0.01|8|10|cosine|0.2|8x4+9=41|
|exp13|7|0.15|2e-5|0.01|8|10|cosine|0.2|8x4+9=41|

DeBERTa-v3-large
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp14|1|0.1|2e-5|0.01|4|10|cosine|0.2|26x4+30=134|
|exp14|2|0.15|2e-5|0.01|4|10|cosine|0.2|26x4+30=134|

DeBERTa-base
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp13|1|0.1|2e-5|0.01|8|10|cosine|0.2|12.25x4=49|
|exp13|2|0.1|2e-5|0.01|8|20|cosine|0.2|23.75x4=95|
|exp13|3|0.15|2e-5|0.01|8|10|cosine|0.2|12.25x4=49|
|exp13|4|0.15|2e-5|0.01|8|20|cosine|0.2|23.75x4=95|
|exp13|5|0.1|2e-5|0.01|8|10|cosine|0.2|12.25x4+13=62|
|exp13|6|0.15|2e-5|0.01|8|10|cosine|0.2|12.25x4+13=62|

DeBERTa-large
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp14|1|0.1|2e-5|0.01|4|10|cosine|0.2|21.8x4+25.6=94|
|exp14|2|0.15|2e-5|0.01|4|10|cosine|0.2|21.8x4+25.6=94|

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
|exp14|5|0.1|2e-5|0.1|4|10|cosine|0.2|17.25x4+20=89|
|exp14|6|0.15|2e-5|0.1|4|10|cosine|0.2|17.25x4+20=89|

XLNet-base-cased
|branch|id|mr|lr|wd|bs|ep|sceduler|wr|time(min)|
|----|----|----|----|----|----|----|----|----|----|
|exp14|1|0.1|2e-5|0.01|8|10|cosine|0.2|12x4+13=61|
|exp14|2|0.15|2e-5|0.01|8|10|cosine|0.2|12x4+13=61|
