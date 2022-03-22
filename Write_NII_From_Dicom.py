'''
input: target_file_path, source_file_path
func: 将source文件夹下的dcm序列转成nii
'''

import SimpleITK as sitk
import os

def Write_NII_From_Dicom(target_file_path, source_file_path):
        # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(source_file_path)
    # 查看该文件夹下的序列数量
    nb_series = len(series_IDs)
    print(source_file_path, f'have {nb_series} dicom files')

    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[0]代表的是第一个序列的ID
    # 如果不添加series_IDs[0]这个参数，则默认获取第一个序列的所有切片路径
    for i in range(nb_series):
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(source_file_path, series_IDs[i])
        # 新建一个ImageSeriesReader对象
        series_reader = sitk.ImageSeriesReader()
        # 通过之前获取到的序列的切片路径来读取该序列
        series_reader.SetFileNames(series_file_names)

        # 获取该序列对应的3D图像
        image3D = series_reader.Execute()
        # 查看该3D图像的尺寸
        sitk.WriteImage(image3D, target_file_path+f'/{i}.nii')


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    source_path = "../PKUH3/"
    dirs = os.listdir( source_path )
    for dir in dirs:
        file_path = source_path+dir
        if not os.path.exists(dir):
            os.mkdir(dir)
        Write_NII_From_Dicom(target_file_path=dir, source_file_path=file_path)


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
