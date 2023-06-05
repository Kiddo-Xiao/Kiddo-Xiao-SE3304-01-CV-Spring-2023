# # 导入所需的库
# import datetime

# # 创建报告内容
# report_title = "实验报告"
# current_date = datetime.datetime.now().strftime("%Y-%m-%d")
# report_content = """
# 报告标题: {}
# 日期: {}

# 超声描述：
# 这是一个示例报告的内容。
# 可以在这里添加详细的实验结果、数据分析、结论等。
# """.format(report_title, current_date)

# # 将报告内容写入文件
# report_filename = "实验报告_{}.txt".format(current_date)
# with open(report_filename, "w") as file:
#     file.write(report_content)

# # 打印生成报告的消息
# print("成功生成报告：{}".format(report_filename))

# 导入所需的库
import datetime

# 创建报告内容
report_title = "实验报告"
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
important_content = "这是重点内容，需要突出显示。"
report_content = f"""
<p> 报告标题: {report_title}
<p> 日期: {current_date}
<p>
超声报告描述：
<p>
这是一个示例报告的内容。
可以在这里添加详细的实验结果、数据分析、结论等。

<p>
重点内容：
<span style="color:red">
<b>{important_content}</b></span>
"""

# 将报告内容写入文件
report_filename = f"实验报告_{current_date}.html"
with open(report_filename, "w") as file:
    file.write(report_content)

# 打印生成报告的消息
print(f"成功生成报告：{report_filename}")
