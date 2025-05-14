import os

def get_image_files(directory):
    # 定义常见的图像文件后缀
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = []

    # 遍历指定文件夹中的所有文件
    for root, dirs, files in os.walk(directory):
        print(files)
        for file in files:
            # 如果文件的后缀在图像后缀列表中，则添加到图像文件列表中
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return image_files

# 示例用法
directory = 'front'  # 将此路径替换为实际的文件夹路径
image_files = get_image_files(directory)
print(image_files)
