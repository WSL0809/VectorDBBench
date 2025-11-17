'''
本文件用于演示 dingodb 向量操作 sdk 的使用方法
'''

# 使用标量过滤
import numpy as np
import os

from dingodb import SDKVectorDingoDB, SDKClient
from dingodb.common.vector_rep import ScalarType, ScalarColumn, ScalarSchema

addrs = "172.30.14.172:22001,172.30.17.173:22001,172.30.17.174:22001"
sdk_client = SDKClient(addrs)
x = SDKVectorDingoDB(sdk_client)
print(x)

index_name = "test_scala_filter"
scheme =  ScalarSchema()
col = ScalarColumn("id", ScalarType.INT64, True)
scheme.add_scalar_column(col)
col = ScalarColumn("name", ScalarType.STRING, False)
scheme.add_scalar_column(col)

x.create_index_with_schema(index_name, 6, scheme, "hnsw", "euclidean", 3, index_config={"efConstruction": 300,"maxElements": 60000,"nlinks": 64}, operand=[5,10,15,20], enable_scalar_speed_up_with_document=True)

d = 16                           # dimension
nb = 15                      # database size
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
print(xb.shape)
xb[:, 0] += np.arange(nb) 
print(xb)
print(xb.shape)

# Dingodb 的 id 只能从 1 开始
ids = [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15]
datas = [{"id": 50}, {"id": 120}, {"id": 130}, {"id": 4.40},{"id": 5.50}, {"id": 6.60}, {"id": 7.70}, {"id": 8.80}, {"id": 9.90}, {"id": 10.10}, {"id": 11.11}, {"id": 12.12}, {"id": 13.13}, {"id": 14.14}, {"id": 15.15}]
vectors = xb.tolist()

for i in range(10):
    x.vector_add(index_name, datas, vectors, ids)

# query_string 为 tantivy 语法
x.vector_search(index_name, vectors[0],is_scalar_speed_up_with_document=True,query_string="id:>=60")

