# ---encoding:utf-8---
# @Time : 2021.02.26
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : util_log.py


import time
import os


class LOG():
    def __init__(self):

        if not os.path.exists('log'):
            os.mkdir('log')
        self.log = open('log/%s.txt' % time.strftime("%Y_%m_%d_%I_%M_%S"), 'w+')
        log_fils = os.listdir('log/')
        log_fils.sort()
        if len(log_fils) > 200:
            print('log file >200,delere old file', log_fils.pop(0), file=self.log)

    def Info(self, *data):
        msg = time.strftime("%Y-%m-%d_%I:%M:%S") + " INFO:"
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + str(info)
        print(msg)
        # print >>self.log,msg
        print(msg, file=self.log)

    def Warn(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " WARN:"
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + info
        # print >> self.log, msg
        print(msg, file=self.log)
        print(msg)

    def Error(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " ERROR:"
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + info
        # print >> self.log, msg
        print(msg, file=self.log)
