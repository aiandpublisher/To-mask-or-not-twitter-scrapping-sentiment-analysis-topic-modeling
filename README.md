# To-mask-or-not-twitter-scrapping-sentiment-analysis-topic-modeling
- It contains codes for data collection, preprocessing, sentiment analysis and topic modeling.
The codes were created for a master dissertation exploring common concerns and public attitudes towards face masks during the pandemic.
# 毕设项目：通过情感分析和话题模型研究推特上英国人对口罩态度的影响因素
### 涉及技术：
推特数据挖掘、机器学习NLP（情感分析、LDA话题模型）、可视化
### 介绍：
疫情下英国人对口罩态度从极度抵制到普遍接受，本项目意在通过挖掘社交媒体数据，研究影响态度的可能因素。
通过与印度数据的横向对比和自身情感、话题历史发展规律分析，定位四大心理因素与文化情境。
### 过程：
- 使用Jira进行项目规划与排期
- 自学社交媒体数据挖掘技术，使用Tweepy连接推特API V2，收集45个口罩相关话题下全部推文，共20000+
- 对数据进行清洗与预处理
- 使用改良后VADER情感分析包分析英国人在相关话题的情感变化趋势，及积极、消极话题数量分别变化趋势，抽取100条推特并手动标记情感作为参照组，VADER算法准确度高达80%
- 使用gensim的LDA主题模型提炼主要话题，通过上百次实验调试模型参数以优化算法效果，使用pyLDAviz可视化话题结果，模型困惑度低至-8.04，语义一致性为0.44
- 使用Health Belief Model心理学理论，结合情感分析结果解读话题，与印度进行对比使得分析更具普遍性。
