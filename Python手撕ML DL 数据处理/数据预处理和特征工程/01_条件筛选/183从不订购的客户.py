"""
183. 从不订购的客户

Customers 表：
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| name        | varchar |
+-------------+---------+
在 SQL 中，id 是该表的主键。
该表的每一行都表示客户的 ID 和名称。

Orders 表：
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| customerId  | int  |
+-------------+------+
在 SQL 中，id 是该表的主键。
customerId 是 Customers 表中 ID 的外键（Pandas 中的连接键）。
该表的每一行都表示订单的 ID 和订购该订单的客户的 ID。

题目要求：
找出所有从不点任何东西的顾客。
以任意顺序返回结果表。

示例 1：
输入：
Customers table:
+----+-------+
| id | name  |
+----+-------+
| 1  | Joe   |
| 2  | Henry |
| 3  | Sam   |
| 4  | Max   |
+----+-------+

Orders table:
+----+------------+
| id | customerId |
+----+------------+
| 1  | 3          |
| 2  | 1          |
+----+------------+

输出：
+-----------+
| Customers |
+-----------+
| Henry     |
| Max       |
+-----------+
"""
#Customers表:id,name   Orders表:id,customer_id
#目标:找出在Orders表中未出现的客户 输出列名为Customers
import pandas as pd
def solve(customers:pd.DataFrame,orders:pd.DataFrame)->pd.DataFrame:
    #过滤:找出id 不在orders['customerId']中的用户
    mask = ~customers['id'].isin(orders['customerId'])
    filtered_df = customers[mask]
    #投影与重命名:选取name列并转换成目标格式
    result = filtered_df[['name']].rename(columns={'name':'Customers'})
    return result


if __name__ == '__main__':
    # 构造题目给的示例数据
    customers_data = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Joe', 'Henry', 'Sam', 'Max']
    })

    orders_data = pd.DataFrame({
        'id': [1, 2],
        'customerId': [3, 1]
    })
    # 运行并打印结果
    ans = solve(customers_data, orders_data)
    print("----- 测试结果 -----")
    print(ans)



