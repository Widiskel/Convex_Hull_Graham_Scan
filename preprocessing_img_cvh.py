import cv2
import os
import numpy as np


def cvh(pts):
  def get_slope(p1, p2): #menghitung lereng
    if p1[0] == p2[0]:
        return float('inf')
    else:
        return 1.0*(p1[1]-p2[1])/(p1[0]-p2[0])
  def cmp(a, b):
        return int(a > b) - int(a < b)
  def turn(p1, p2, p3): #cek rotasi , jika hasil negatif maka rotasinya kekanan dan harus dihapus dari stack
        return cmp((p2[0] - p1[0])*(p3[1] - p1[1]) - (p3[0] - p1[0])*(p2[1] - p1[1]), 0)

  hull=[]
  pts = sorted(pts)
  start = pts.pop(0) #pick starting point
  pts.sort(key=lambda p: (get_slope(p,start), -p[1],p[0])) #urutkan sisa poin dengan cara mengurutkan hasil perhitungan lereng(slope) antara point dengan starting point
  hull.append(start) #menambahkan starting point ke stack
  for x in range(len(pts)):
    hull.append(pts[x]) #menambahkan point ke stack hull
    while len(hull) > 2 and turn(hull[-3],hull[-2],hull[-1]) != 1: #jika stack sudah lebih dari 3 maka akan dilakukan pengecekan setiap triplet poin dari titik saat ini ke dua titik sebelumnya pada stack 
      hull.pop(-2)

  return hull

def pp(diri,diro):
  os.chdir("/content/drive/MyDrive/Skenario 4")
  directory = diri
  outdirectory = diro
  dirlist = os.listdir(directory)
  for x in dirlist:
      y = os.path.join(directory,x)
      dirlist2 = os.listdir(y)
      for z in dirlist2:
          path = os.path.join(y,z)
          outpath = os.path.join(outdirectory,x)
          outpath2 = os.path.join(outpath,z)
          img = cv2.imread(path)
          
          img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          blur = cv2.GaussianBlur(img_gray,(1,1),0)
          canny_output = cv2.Canny(blur, 40, 255)
          # Find contours
          contours, hierarchy = cv2.findContours(canny_output,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
          arr =[]
          for i in range(len(contours)):
              ca = []
              for j in range(len(contours[i])):
                  ca.append([contours[i][j][0][0], contours[i][j][0][1]])
              arr.append(ca)

          hull=[]
          
          for i in range(len(arr)):
            hull.append(np.array(cvh(arr[i])))

          # create an empty black image
          drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
          # draw contours and hull points
          
          cv2.drawContours(drawing, hull, -1, (255,0,0), 2, 8)
          cv2.drawContours(drawing, contours, -1, (0,255,0), -1)
                
          if(outdirectory not in os.listdir()):
              os.mkdir(outdirectory)
          if(x not in os.listdir(outdirectory)):
              os.mkdir(outpath)
          cv2.imwrite(outpath2, drawing)

print("->Preprocessing Training")
pp("Training","TrainingPP")
print("->Preprocessing Testing ")
pp("Testing","TestingPP")

print("->Preprocessing Complete ")
