# Project Searcher
This project use machine learning technique to help you to search what you want in the project you want to read.

## Installation
```shell
git clone https://github.com/Rota1001/ProjectSearcher.git
cd ProjectSearcher
pip3 install -r requirements.txt
```
If you want to use gpu, please set up your Pytorch environment yourself, or you can just use cpu version as well.

## 使用方法
### 開啟UI介面
```shell
python3 ui.py
```
第一次開啟的時候要下載模型會比較久

### 預處理專案
- Load File

    在`Load File`頁面，輸入專案位置，按下`Load File`鍵
- Load Weights

    一般來說Load File之後會自動載入權重，但是如果想載入自己之前生出來的權重也行。把`comment.pkl`和`data.pkl`放在`data/`之下(沒有的話自己建一個)，然後到`Load File`頁面，按下`Load Weights`鍵即可

### 搜尋
- Search
  
  輸入要搜尋的東西的描述，按下`Search`鍵即可進行搜尋
- 開啟檔案

  搜尋結果是按鈕，按下後會在VSCode裡打開。如果沒辦法使用，請在環境變數中加入VSCode。
  
