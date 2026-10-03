"""
1148. 文章浏览 I
难度：简单

【表结构：Views】
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| article_id    | int     |
| author_id     | int     |
| viewer_id     | int     |
| view_date     | date    |
+---------------+---------+
此表可能会存在重复行（在 SQL 中无主键）。
每一行表示某人在某天浏览了某位作者的某篇文章。
注意：同一人的 author_id 和 viewer_id 是相同的。

【题目要求】
请查询出所有浏览过自己文章的作者。
结果需要按照作者的 id 升序排列。

【示例】
输入（Views 表）：
+------------+-----------+-----------+------------+
| article_id | author_id | viewer_id | view_date  |
+------------+-----------+-----------+------------+
| 1          | 3         | 5         | 2019-08-01 |
| 1          | 3         | 6         | 2019-08-02 |
| 2          | 7         | 7         | 2019-08-01 |
| 2          | 7         | 6         | 2019-08-02 |
| 4          | 7         | 1         | 2019-07-22 |
| 3          | 4         | 4         | 2019-07-21 |
| 3          | 4         | 4         | 2019-07-21 |
+------------+-----------+-----------+------------+

输出：
+------+
| id   |
+------+
| 4    |
| 7    |
+------+
"""
import pandas as pd
#浏览过自己文章: author_id == viewer_id
#去重:pd.DataFrame.drop_duplicates(keep='first')->pd.DataFrame  默认整行所有列完全相同时只保留一条
#排序 pd.DataFrame.sort_values(by='id',ascending=True)->pd.DataFrame

def article_views(views:pd.DataFrame)->pd.DataFrame:
    #条件筛选 作者==浏览者 正向筛选不用~
    mask = views['author_id'] == views['viewer_id']
    filtered_df = views[mask]
    #投影与改名:只留author_id那一列 且输出改名为id
    df = filtered_df[['author_id']].rename(columns={'author_id':'id'})
    result = df.drop_duplicates().sort_values(by='id')
    return result

if __name__ == '__main__':
    # 构造题目给的测试数据
    data = [
        [1, 3, 5, '2019-08-01'],
        [1, 3, 6, '2019-08-02'],
        [2, 7, 7, '2019-08-01'],  # 7号看了自己
        [2, 7, 6, '2019-08-02'],
        [4, 7, 1, '2019-07-22'],
        [3, 4, 4, '2019-07-21'],  # 4号看了自己
        [3, 4, 4, '2019-07-21']   # 4号重复看自己
    ]
    columns = ['article_id', 'author_id', 'viewer_id', 'view_date']
    views_df = pd.DataFrame(data, columns=columns)
    # 运行并打印
    ans = article_views(views_df)
    print("----- 测试结果 -----")
    print(ans)
