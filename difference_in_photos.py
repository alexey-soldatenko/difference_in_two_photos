import numpy as np
import cv2
from matplotlib import pyplot as plt

#читаем значение пикселей из фото в оттенках серого
img1 = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

#высчитываем разрешение фото
count_y = 0
count_x = 0
for i in img1:
	count_y += 1
	count_x = len(i)

#методом наименьших квадратов определяем линейную функцию
x = np.sum(img1)
y = np.sum(img2)
double_x = np.sum(img1 ** 2)
x_y = np.sum(img1*img2)

#определяем систему линейных уравнений
a = np.array([[double_x, x], [x, count_y]])
b = np.array([x_y, y])

#получаем результат решения системы линейных уравнений
result = np.linalg.solve(a, b)
#значения линейной функции
line = np.array(result[0]*img1+result[1])


#определяем наиболее удаленные точки от линейной функции
new2 = img2[(abs(img2-line) > 30)]
new1 = img1[(abs(img2-line) > 30)]

'''
#выводим полученные данные
plt.scatter(img2, img1)
plt.plot(img1, line)
plt.grid()
plt.figure()


plt.scatter(new2, new1)
plt.grid()

plt.show()
'''

#определяем шаблон нового фото (разницы между двумя первыми)
out = np.zeros((count_y, count_x))

for y in range(len(img2)):
	#перебираем пиксели оригинального фото
	for x in range(len(img2[y])):
		#если происходит совпадение по оси х, 
		#проверяем совпадение по оси у
		if img2[y][x] in new2:
			v = img1[y][x]
			index = np.where(new2 == img2[y][x])
			if v in new1[index]:
				#при полном совпадении вносим изменение в выходное фото
				out[y][x] = 255

#записываем полученное фото
cv2.imwrite('out.jpg', out)
f_out = cv2.imread('out.jpg', cv2.IMREAD_GRAYSCALE)

#выполняем фильтрацию методом эрозии
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(f_out,kernel,iterations = 1)

#показываем полученное фото
cv2.imshow('image',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()


