import matplotlib.pyplot as plt
from proplot import rc

# 数据列表
data = [
    ("Arts and Entertainment", 159),
    ("Computers and Electronics", 153),
    ("Education and Communications", 150),
    ("Food and Entertaining", 153),
    ("Health", 159),
    ("Hobbies and Crafts", 159),
    ("Holidays and Traditions", 159),
    ("Home and Garden", 159),
    ("Sports and Fitness", 159),
    ("Travel", 159)
]
rc["font.family"] = "Times New Roman"

# 分离类别和对应的值
labels, values = zip(*data)

# 设置颜色
colors = [
     '#EF4F4FFF', '#FA7070FF', '#7AA2E3FF', '#52D3D8FF', '#FFE699FF', '#EF4F4FBF', '#FA7070BF', '#7AA2E3BF', '#52D3D8BF', '#FFE699bF'
]

# 绘制饼状图
plt.figure(figsize=(17, 10))
wedges, texts, autotexts = plt.pie(
    values,
    labels=labels,
    colors=colors,
    autopct=lambda p: '{:.0f}'.format(p * sum(values) / 100),  # 显示数量
    startangle=140
)

# 设置标签字体大小
for text in texts:
    text.set_fontsize(50)

# 设置数量标签字体大小
for autotext in autotexts:
    autotext.set_fontsize(50)
    autotext.set_color("black")

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# 添加标题
# plt.title('Category Distribution')
# 显示图表
plt.show()

# 显示图表
plt.savefig('utils/graph/pie_chart.svg')
