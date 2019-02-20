# Img_recg_ali_wechat
The sklearn deep learning framework training model is used to identify Alipay and WeChat payment code.(通过使用sklearn深度学习框架训练模型,来识别支付宝,微信收款码)



### 项目结构

```json
├─config              # 配置文件                                                   
├─img_recognition     # 主程序目录      
│  ├─data             # data模块, 存放训练图片,测试图片       
│  │  ├─alipay        # 使用alIpay收款手机截图作为测试图片        
│  │  │  ├─test       # 测试图片目录      
│  │  │  │  └─receive # 测试识别收款条目      
│  │  │  └─train            
│  │  │      └─receive      
│  │  └─wechat        # 微信支付手机截图测试      
│  │      ├─test      # 测试分类      
│  │      │  ├─money  # 测试金额      
│  │      │  ├─reason # 测试收款原因      
│  │      │  └─receive# 测试收款备注     
│  │      └─train     # 训练模型原始图片      
│  │          ├─money       
│  │          ├─reason      
│  │          └─receive     
│  ├─log                   #日志目录 
│  └─model                 # 训练后保存的模型
|  |-train.py 	           # 训练主函数
└─opencv_test              # 结合opencv图像处理,检查sklearn正确率
```

​        

### update



​       