# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file: remoteKeyboard
@time: 2020.03.25
"""

from bottle import run, route, request, redirect, error, ServerAdapter
import socket
import ui
import keyboard


# 该函数用于获取本地ip地址
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('google.com', 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = 'N/A'
    return ip


# 主页，用于用户访问输入。
# 目前demo中的前端实现非常简单，仅有单行提交的功能，多行提交和实时异步提交等功能将会是以后的发展方向。

def index():
    return '''
		<form action="/" method="post">
			请输入：<input name="input" type="text" />
			<input value="发送" type="submit" />
		</form>
		'''


# 当用户提交数据时调用
# 将来可在此对以不同方式提交的数据做各种操作，由用户自由发挥

def input():
    inputdata = request.forms.getunicode('input')  # 以unicode编码获取提交的数据，否则中文将会是乱码
    print(inputdata)  # 输出以便在控制台调试时观察
    if keyboard.is_keyboard():  # 判断是否是在键盘中运行
        keyboard.insert_text(inputdata)  # 将提交的数据插入当前光标位置
    return redirect('/')  # 重定向到输入页面，今后将前端页面的数据提交以Ajax异步提交来实现以后可省去重定向


# 当用户输入错误网址时调用

def error404(error):
    return '对不起，这里什么也没有...<br/><a href="/">返回主页</a>'


# 单独实现一个我们自己的server类，主要是为了实现stop方法
class MyWSGIRefServer(ServerAdapter):
    server = None

    def run(self, handler):
        from wsgiref.simple_server import make_server, WSGIRequestHandler
        if self.quiet:
            class QuietHandler(WSGIRequestHandler):
                def log_request(*arg, **kw): pass

            self.options['handler_class'] = QuietHandler
        self.server = make_server(self.host, self.port, handler, **self.options)
        self.server.serve_forever()

    # 该方法用于在主窗口线程中调用，以停止服务器线程
    def stop(self):
        print('server stop')
        self.server.shutdown()


class MainWindow(ui.View):

    def __init__(self, server, localIP, port):
        self.server = server

        self.name = 'Keyboard Preview'  # 仅在调试窗口中才看到窗口名
        self.flex = 'WHTBLF'  # 尺寸边距都设为自动，以便填充满整个键盘

        # 创建iplabel以显示当前服务器运行的ip及端口
        iplabel = ui.Label()
        iplabel.flex = 'WHTBLF'
        iplabel.text = "请在浏览器打开：" + localIP + ":" + str(port)
        iplabel.background_color = "#ffffff"
        iplabel.alignment = ui.ALIGN_CENTER

        # 之后还可以继续开发更多组件件加入主窗口
        self.add_subview(iplabel)

        # 判断是否是在键盘中运行
        if keyboard.is_keyboard():
            keyboard.set_view(self, 'expanded')
        else:
            # 当在pythonista主应用中调试时启动:
            self.frame = (0, 0, 500, 200)
            self.present('sheet')

    # 在窗口关闭时停止服务
    def will_close(self):
        self.server.stop()


def main():
    localIP = get_local_ip()  # 本地ip地址
    port = 2333  # 服务器端口，用户可自行设置
    server = MyWSGIRefServer(host=localIP, port=port)  # 创建服务

    v = MainWindow(server, localIP, port)  # 创建主窗口

    run(server=server)  # 运行服务，这一步要在最后启动，不然会阻塞线程


if __name__ == '__main__':
    main()
