# SIGNATE Student Cup 2022

## 振り返り
全体としての感想
- 初めてのコンペであり，初めての言語データであったため，コンペの流れや言語データの扱い方など学べる点が多かった．
- CV, LB, Public LB がほとんど相関せず，コンペとしては難しかった．small data である上，F1 macro で評価されるため，small sample の class の影響によって大きくスコアが変動したためと考えられる．

取り組んだこと
- 前処理
    - HTML タグの除去（bs4）
    - u202f や https など不必要な情報の置換
- データ拡張
    - mask token へのランダム置換(他の DA 手法は実装はしたものの影響評価できず)
    - nlpaugs による類義語置換
    - 箇条書きアイテムの順番入れ替え
- モデル
    - CV が良さげな roberta-large, deberta-v3-large に最後は絞り込んだ
    - roberta-{base, large}
    - deberta-{base, large}
    - deberta-v3-{base, large}
    - xlnet-{base,large}-cased
- 損失関数
    - Focal Loss：効果あり
    - Weighted Cross-Entropy Loss：あまり効果なし
    - Dice Loss：学習が難しい
- 学習方式
    - Adversarial Weight Perturbation：roberta 系統などで効果あり（eps=1e-3 fixed が良さげ）
    - Fast Gradient Method：AWP を上回る良さはなし
    - Masked Language Model による事前学習（前処理が間違えている状態のため，影響評価できず）
- その他
    - pytorch ligthning によるラッパー
    - CV で統一して同じ experiment として comet logging
    
今後の課題・反省点
- 前処理
    - ステム・レマタイズ：a, the, of などが言語モデルでどれぐらい効くか不透明だったため適用を見送り
    - 大文字・小文字の調整
    - 語彙の追加：「python」はすでに登録されていた．登場回数が少ない TensorFlow, PyTorch　など鍵となるフレームワークやソフトウェアの名前が多数登場．時間の都合で追加した場合の影響を評価できなかった．
    - 入念なデバッグ：最終日手前で前処理のミスが発覚．前処理のデバッグを notebook 上では行っていたが実機では行っていなかった．
- データ拡張
    - アイテムの入れ替え・削除・特殊トークンの挿入など，箇条書きであるということを活かしたかった
    - mixup
- 学習方式
    - 集団事例学習：アイテムごとや文ごとの予測結果を aggregate する形式を試してみたかった．文章全体とは異なる要素を抽出する可能性がある．
    - 疑似ラベリング：ルールで推論時間を気にされていたので導入を見送り．kaggler 御用達みたいなので今後試してみたい．
    - ラベル平滑化：ラベルミスがない場合でも効果を発揮するのか，統計的にちゃんと考えたい．
    - ラベル遷移確率：valid confusion matrix から推定可能．導入見送りはラベル平滑化と同様
- 時間管理
    - （コロナで倒れていたせいであるが）取り組み始めるのが遅かった．もう少し早めに取り組みたかった．
    - 1日の投稿回数上限を最終日まで知らなかった．ルール上に一切記載がなかったためであるが，序盤に提出回数上限を試して調べてみても良かった．
    - GPU 有効活用のため，job queue のシステムを作りたかった．grid search のコードは作ったが，前の grid search が終わり次第，次の grid search に行くなど，queue があれば便利
    - fold0 での early stopping を検討しても良かった．1つの fold で結果がイマイチなものは次に進めない，など
- 描画・記録
    - 判別難易度が高い文章の抽出
    - アテンションの可視化：どの単語が分類に効いているか
    - 判別ミスしている文章を羅列して特徴を模索
- コード
    - 途中で知ったのだが，kaggler はブランチを切るのではなく実験ごとにコードを生成する．確かにその方が再現性担保しやすい．実験ごとのディレクトリを自動生成して，現在のファイルを全てコピーするような仕組みを作っても良かった．


## score
最後の結果を提出し，PLB 0.726（最終58位,blonze）でフィニッシュ

|branch|CV|LB|PLB|rdir|hparams|
|----|----|----|----|----|
|exp02|0.717|0.748|0.731|12-roberta-base2|ep=20, lr=2e-5, do=0.1, FL(2), wd=0.1, sc=None|
|exp9|0.724|0.745|0.718|17-deberta-v3-large1|ep=10, mr=0.1, lr=2e-5, do=0.1, FL(2), wd=0.01, gc=1, sc=Linear(ws=50)|
|exp10|0.695|0.714|0.709|17-deberta-base1|ep=10, mr=0.1, lr=2e-5, do=0.1, FL(2), wd=0.01, gc=1, sc=Linear(ws=50), at=AWP(eps=1e-2,lr=1e-4)|
|exp15|0.717|0.718|0.737|19_gs1/mnroberta-basebs16wd0.1mi5umFalselFLg2aawpal1.0g0|ep=10, mr=0,1, lr=2e-5, do=0.1, sc=Linear(ws=100)|
|exp15|0.715|0.710|0.729|19_gs1/mnxlnet-base-casedbs16wd0.01mi1umFalselFLg2aawpal1.0g0|ep=10, mr=0.1, lr=2e-5, do=0.1, sc=Linear(ws=100)|
|exp16|0.692|0.729|0.706|22_gs1/mnmicrosoft/deberta-v3-largebs8wd0.01e10aNoneal1.0g2|mr=0.1, lr=2e-5, do=0.1, sc=Linear(ws=100)|
|exp16|0.716|0.749|0.725|22_gs1/mnroberta-largebs8wd0.1e10mi5umTruel2e-05g0|mr=0.1, do=0.1, sc=Linear(ws=100)|
|exp16|0.722|0.726|0.728|23_gs1/mnmicrosoft/deberta-v3-largebs8wd0.01e10mi1mr0.1afgmal0.1g0|do=0.1, sc=Linear(ws=100)|
|exp17|0.711|0.734|0.720|24_gs1/mnroberta-largebs8wd0.1e10mi5lFLaawpal1.0g1|do=0.1, sc=Linear(ws=100)|
|exp17|0.710|0.731|0.703|24_gs1/mnmicrosoft/deberta-v3-largebs8wd0.01e10mi1lFLaNoneal1.0g2|do=0.1, sc=Linear(100)|
|exp18|0.740|0.752|0.726|25_esm_f1_Nelder-Mead1|ensemble of 2 models|

## MLM pretrained
exp8~16 が前処理で重大な欠陥（別のデータを挿入してしまっている）があるブランチのため，今回は影響評価できず
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
