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
