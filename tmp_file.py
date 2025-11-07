'''
本文件用于演示 dingodb 向量操作 sdk 的使用方法
'''
import numpy as np
import os

from dingodb import SDKVectorDingoDB, SDKClient
from dingodb.common.vector_rep import ScalarType, ScalarColumn, ScalarSchema

addrs = "172.30.14.172:22001,172.30.14.173:22001,172.30.14.174:22001"
sdk_client = SDKClient(addrs)
x = SDKVectorDingoDB(sdk_client)

index_name = "test_index"

x.create_index(index_name, 6)


# x.delete_index(index_name)
# breakpoint()

# 构建训练数据和查询数据
d = 6                           # dimension
nb = 4                      # database size
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')

xb[:, 0] += np.arange(nb) / 1000.


ids = [1, 2, 3, 4]
datas = [{"a1": "a"}, {"a2": 1}, {"a3": 2.2}, {"a4": True}] # 标量只支持这4种类型
vectors = xb.tolist()

res = x.vector_add(index_name, datas, vectors, ids)
print(res)

res = x.vector_search(index_name, vectors, 2)
print(res)
res = x.vector_search(index_name, vectors, search_params={"langchain_expr": "eq('a3',2.1)"})  
print(res)
x.delete_index(index_name)
