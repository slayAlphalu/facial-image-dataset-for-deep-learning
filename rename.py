import os
import xlrd


data = xlrd.open_workbook('./rating.xlsx')
path = './web_image'
data = data.sheet_by_index(0)
col_id = data.col_values(0)
col_score = data.col_values(1)

for image in os.listdir(path):
    if (image == ".DS_Store"):
        continue
    if "-" in image:
        print(image)
        print("find -")
        continue
    
    id=image[7:10]
    print(id)    
    for i in range(1,683):
       if int(id)==int(col_id[i]):
           print('done')
           score=int(col_score[i])
           print(score)
           os.rename(os.path.join(path,image),os.path.join(path,str(score)+'-'+id+".jpg"))
 
   
