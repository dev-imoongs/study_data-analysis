import pandas as pd

df = pd.read_csv('./datasets/Global_Superstore2.csv', encoding='CP1252')
df

# 1. 결측치 검사
df.info()

# 2. 중복값 검사
df.duplicated().sum()

### RFM

analysis_date = '2023-10-06'

# 구매일자로부터 지금의 날짜를 연산하기 위한 식
order_date = pd.to_datetime(df['Order Date'],dayfirst=True)
df['Recency'] = (pd.to_datetime(analysis_date, dayfirst=False) - order_date).dt.days
df['Monetary'] = df['Sales'] * df['Quantity']

df_rfm = df.groupby('Customer ID').agg({'Recency': 'min', 'Customer ID': 'count', 'Monetary': 'sum'}).rename(columns={'Customer ID': 'Frequency'})

# Customer ID로 groupby 했기 때문에 행 개수가 달라져 reset_index 사용
df_rfm = df_rfm.reset_index()
df_rfm

from sklearn.preprocessing import MinMaxScaler

normalization = MinMaxScaler()
rfm_normalization = normalization.fit_transform(df_rfm[['Recency', 'Frequency', 'Monetary']])
rfm_normalization = pd.DataFrame(rfm_normalization, columns=['Recency', 'Frequency', 'Monetary'])

df_rfm[['Frequency', 'Recency', 'Monetary']] = rfm_normalization[['Frequency', 'Recency', 'Monetary']]
df_rfm

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering()

# 실루엣 점수 확인 위해 K-elbow 시각화
model = KElbowVisualizer(model, k=(2,10), timings=False)
model.fit(df_rfm.iloc[:,1:])

model.show()

agloCluster = AgglomerativeClustering(n_clusters=4).fit(rfm_normalization)

df_rfm['cluster'] = agloCluster.labels_
print(df_rfm['cluster'].value_counts())

df_rfm

import matplotlib.pyplot as plt
import seaborn as sns

# Recency : 얼마나 최근에 구매했는가
# Frequency : 얼마나 자주 구매했는가
# Monetary : 얼마나 많은 금액을 지출했는가
titles = ['Recency', 'Frequency', 'Monetary']

# 집단 개수
k = 4

# 각 항목별
for title in titles:
    plt.figure(figsize = (5, 5))
    
#     집단 별
    for i in range(k):
#         scatter: 산점도(분포도)
        plt.scatter(df_rfm.loc[df_rfm['cluster'] == i, 'cluster'], 
                    df_rfm.loc[df_rfm['cluster'] == i, title],
                    label = f'cluster {i}')
    
#     색상별 제목(label) 표시
    plt.legend()
    plt.title(title, size = 15)
    plt.xlabel('cluster', size = 12)
    plt.ylabel(title, size = 12)
    plt.show()

titles = ['Recency', 'Frequency', 'Monetary']

for title in titles:
    plt.figure(figsize = (5, 5))
    sns.boxplot(x = df_rfm.cluster, y = df_rfm[title], palette='muted')
    plt.title(title)
    plt.show()

# 이상치 조정
condition1 = df_rfm['cluster'] == 3
condition2 = df_rfm['Recency'] >= 0.7

condition3 = df_rfm['cluster'] == 0
condition4 = df_rfm['Frequency'] >= 0.9

condition5 = df_rfm['Monetary'] >= 0.75

df_rfm.loc[condition1 & condition2, 'Recency'] = df_rfm[condition1]['Recency'].median()
df_rfm.loc[condition3 & condition4, 'Frequency'] = df_rfm[condition3]['Frequency'].median()
df_rfm.loc[condition3 & condition5, 'Monetary'] = df_rfm[condition3]['Monetary'].median()

for title in titles:
    plt.figure(figsize = (5, 5))
    sns.boxplot(x = df_rfm.cluster, y = df_rfm[title], palette='muted')
    plt.title(title)
    plt.show()

# 가중치를 매기기 위한 클러스터 mean값 확인
df_rfm_cluster0_mean = df_rfm[df_rfm['cluster'] == 0]['Recency'].mean()
df_rfm_cluster1_mean = df_rfm[df_rfm['cluster'] == 1]['Recency'].mean()

print(df_rfm_cluster0_mean, df_rfm_cluster1_mean)

Cluster 0: VIP, (1, 4, 4) = 9  
Cluster 1: Silver, (3, 2, 2) = 7  
Cluster 2: Gold, (2, 3, 3) = 8  
Cluster 3: Bronze, (4, 1, 1) = 6

df_rfm['cluster'] = df_rfm['cluster'].replace([0, 1, 2, 3], ['VIP', 'Silver', 'Gold', 'Bronze'])
df_rfm

order = ['Silver', 'Gold', 'VIP', 'Bronze']
# create a countplot
print('Cluster Count:')
print(df_rfm['cluster'].value_counts())

sns.countplot(x='cluster', data=df_rfm, palette='muted', order=order)
plt.title('Clusters')
plt.show()
