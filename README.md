  参数说明：
S=1              将原图缩放至1倍
sigma=[1.0]        使用高斯核平滑初始图像，标准偏差1.0   
kx=[FAST]          	 用FAST提取特征点
     =SIFT            用SIFT提取特征点
     =DENSE_SAMPLING 用密集采样提取特征点            
   ds_spacing=[10]    密集采样的空间
   gm=NONE            关闭几何同型验证                     
     =AFFINE          仿射仿射几何验证
     =[HOMOGRAPHY]    同形几何验证                 
   gv_dist=[16]       几何验证距离（如果启用）
   FAST_thresh=[80]   使用阈值80进行快速关键点提取。（值越小提取的点越多）
   L=[8]              创建一个8层的图像金字塔
   q0=0.1            从条带的0.1处开始编码
   q1=0.8            从条带的0.8处截止编码
   nS=[9]             把条带分为9段
   b=[2]              每个段都用两位来编码
   nL=[20]            限制哈希表的容量
   om=matches.jpg     把匹配结果输出至图片
   nM=40             只输出最大的40个匹配
  
   vis=[LINES]        用线来进行匹配可视化
      =MESHES      用网格来进行匹配可视化

使用方法：
如果修改之后进行编译推荐在64位release环境下运行
打开D-nets项目下x64/Release/,D-nets编译完成的程序在这里
然后打开cmd，然后输入D-nets.exe的路径，后面跟上参数，具体格式如下
D-Nets路径 图片1路径 s=1 图片2路径 s=1 参数 参数（不需要注意顺序）
常用参数示例：



C:\Users\J1ose\source\repos\D-nets\x64\Release\D-nets.exe C:\Users\J1ose\Desktop\test-dnets\xz(1).jpg s=1 C:\Users\J1ose\Desktop\test-dnets\xz(2).jpg s=1 kx=FAST FAST_thresh=40 vis=MESHES nM=40

然后会出来两个图片的结果和两个点序号的窗口，同时文件会输出到设置的路径（在代码中设置）
