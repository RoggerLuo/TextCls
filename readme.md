# attention encode and filter
在重复一个字的案例中  
比如在区别 “这是圆的分类” 与 “这是圆圆的分类”这两句的任务中，  

filter的attention的softmax结果是 98%\96.9%  

encode的attention的softmax结果是 99.99999%\99.999999%

encode会比filter准确很多，训练速度也会慢一些  