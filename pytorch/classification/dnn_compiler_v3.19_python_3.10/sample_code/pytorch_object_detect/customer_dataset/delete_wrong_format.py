import os

# 设置要处理的目录路径
TXT_FOLDER = r"C:/Users/7000031151/OneDrive - Sony/04_IMX681/Compiler/dnn_compiler_v3.19/dnn_compiler_v3.19/dnn_compiler_v3.19_python_3.10/sample_code/pytorch_object_detect/customer_dataset/validation/labels"  # 这里改成你的txt文件所在路径

# 遍历目录
for file in os.listdir(TXT_FOLDER):
    if not file.endswith(".txt"):
        continue

    txt_path = os.path.join(TXT_FOLDER, file)

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 删除所有逗号
    new_content = content.replace(",", "")

    # 仅当有变化时才写回
    if new_content != content:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✅ 已处理：{file}")

print("🎯 所有txt文件的逗号已删除完成！")
