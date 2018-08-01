import paramiko
# 实例化一个trans对象# 实例化一个transport对象
trans = paramiko.Transport(('180.166.133.28', 9995))
# 建立连接
trans.connect(username='shaoye123', password='123456')
# 实例化一个 sftp对象,指定连接的通道
sftp = paramiko.SFTPClient.from_transport(trans)
# 下载文件
sftp.get(remotepath='/home/zhangxuan/bq_data.csv', localpath='/home/zhangxuan/insert2/bq_data.csv')
trans.close()


