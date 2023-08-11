# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:13:02 2022

@author: Admin
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

# Đọc ảnh
image1 = cv2.imread('X:/Users/Admin/Downloads/project2_DA2/cho2.jpg')
# Chuyển ảnh sang RGB
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# ảnh xám
training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
# xoay ảnh
test_image = cv2.pyrDown(training_image)
test_image = cv2.pyrDown(test_image)
num_rows, num_cols = test_image.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))

test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
# In ra kết quả
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Training Image")
plots[0].imshow(training_image)

plots[1].set_title("Testing Image")
plots[1].imshow(test_image)
# Tìm điểm đặc trưng ở hình ảnh ban đầu
sift = cv2.SIFT_create()
# hàm tìm các điểm đặc trưng
train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)
keypoints_without_size = np.copy(training_image)
keypoints_with_size = np.copy(training_image)
#vẽ các vòng tròn 
cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))
cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# ------
fx, plots = plt.subplots(1, 2, figsize=(20,10))
plots[0].set_title("Train keypoints Without Size")
plots[0].imshow(keypoints_without_size, cmap='gray')

plots[1].set_title("Train keypoints With Size")
plots[1].imshow(keypoints_with_size, cmap='gray')
# In số lượng các điểm chính được phát hiện trong hình ảnh đào tạo
print("Số các điểm hấp dẫn ảnh ban đầu : ", len(train_keypoints))
#Tìm điểm đặc trưng ở hình ảnh sau khi xoay
sift = cv2.SIFT_create()

train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)

keypoints_without_size = np.copy(test_image)
keypoints_with_size = np.copy(test_image)

cv2.drawKeypoints(test_image, test_keypoints, keypoints_without_size, color = (0, 255, 0))

cv2.drawKeypoints(test_image, test_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
####
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Test keypoints Without Size")
plots[0].imshow(keypoints_without_size, cmap='gray')

plots[1].set_title("Test keypoints With Size")
plots[1].imshow(keypoints_with_size, cmap='gray')

# In số lượng các điểm chính được phát hiện trong hình ảnh đào tạo
print("Số các điểm hấp dẫn ảnh xoay : ", len(test_keypoints))
#ĐỐI SÁNH ẢNH
#Cách 1.Tạo một đối tượng Brute Force Matcher.
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
# Tìm ra kết quả đối sánh tốt nhất
matches = bf.match(train_descriptor, test_descriptor)
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(training_image, train_keypoints, test_image, test_keypoints, matches, test_gray, flags = 2)
plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.title('Điểm phù hợp')
plt.imshow(result)
plt.show()
print("\nSố lượng điểm trùng khớp: ", len(matches))
# Cách 2: tìm ra L kết quả đối sánh tốt nhất, trong đó L tùy chỉnh bởi người xử lí
#  theo D_lower thì mặc đinh K=2
matchess = cv2.BFMatcher().knnMatch(train_descriptor, test_descriptor, k=2) 
#Sift áp dụng một ngưỡng mặc định chạy từ 0->1, nên có thể tùy chỉnh để đưa ra kết quả
good = [[m] for m, n in matchess if m.distance < 0.81*n.distance]
img3 = cv2.drawMatchesKnn(training_image, train_keypoints, test_image, test_keypoints, good, None,
                          matchColor=(0, 255, 0), matchesMask=None,
                          singlePointColor=(255, 0, 0), flags=0)

plt.imshow(img3),plt.show()
print("\nSố lượng điểm trùng khớp: ", len(good))