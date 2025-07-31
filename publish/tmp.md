
目前有```(hist, item, label)```，label可以是正样本或者负样本

+ 输入(hist,item)，问模型用户会不会喜欢这个item(pred)，并总结用户偏好(reasoning)
+ 输入(hist,item,reasoning)，问模型这个用户偏好是否合理（judge），并反思出不合理的原因（reflect）
    + 如果judge == 合理 and pred == label：
        + ```[(hist),(reasoning)]```加到**reasoning数据集**里
        + ```[(hist,reasoning), (judge, reflect="")]```加到**reflect数据集**里
    + 如果judge == 不合理 and pred != label：
        + 输入hist,item,reasoning,reflect，让模型输出用户会不会喜欢这个item(pred')和refine后的用户偏好(reasoning_r)
        + 如果pred' == label：
            + ```[(hist,reasoning), (judge, reflect)]```加到**reflect数据集**里
            + ```[(hist,reasoning,reflect),(reasoning_r)]```加到**refine数据集**里
    + else：扔掉```(hist, item, label)```