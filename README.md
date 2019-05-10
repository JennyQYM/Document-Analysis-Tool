# Document-Analysis-Tool

This is a flask application for text folder analysis.

<img src="pics/pic1.jpg" />


## Flask Struct

- models: to store keywords extraction | LDA models
- static: to store the documents folder | wordcloud image | heatmap image | cleaned data
- templates: a collection of html

## Usage

cd to the doc_analysis
```
$ python runserver.py
```

## Output

- documents similarity matrix calculated by the LDA model
- document's wordcloud | keywords | key phrases 
