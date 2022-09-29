import cv2
import numpy as np
import matplotlib.pyplot as plt

# 텍스처 이미지 로드
texture = cv2.imread('img/crack_texture.png', 0)
texture = texture[0:600, 200:800] # size (600, 600)

# 폰트 좌표 추출
src = cv2.imread('img/font_13.png', 0)
_, dst = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
dst = 255 - dst  # inverse

contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digits_coordinate = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    digits_coordinate.append((x, y, w, h))

digits_coordinate.sort(key=lambda x:x[0]) # x좌표 기준으로 정렬

# 폰트 마스크 추출
pad = 3
digits_len = len(digits_coordinate)

# 이미지 사이즈
size = 30

# 배경 생성
bg = np.random.randint(30, size=(size, size), dtype=np.uint8)
bg = bg + 150 # range 150 ~ 180

# 블러 커널, 밝기 낮추는 커널
kernel = np.ones((3,3), np.float32) / 9
base = np.full((size, size), 170, dtype=np.uint8)

output = []

for i in range(digits_len):
    x, y, w, h = digits_coordinate[i]
    # crop
    mask = dst[y-pad:y+h+pad, x-pad:x+w+pad]

    # add padding
    h_pad = (size - h) >> 1
    w_pad = (size - w) >> 1

    # add black border 
    border = cv2.copyMakeBorder(mask, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, None, 0)
   
    # resize
    # 바이너리 인버스
    resized = cv2.resize(border, (size, size))
    
    # 숫자 부분만 밝히기 위해 배경 0 숫자는 50 마스크 만듬
    bright_mask = cv2.subtract(resized, base) # max 0 - 100

    # operation
    sub = cv2.subtract(bg, bright_mask) # 그레이 배경 추가

    # blur
    blur = cv2.filter2D(sub, -1, kernel)
    output.append(blur)

# 배경, 숫자, 텍스처 이미지 연산
    
# 저장

# 보기
for i in range(digits_len):
    cv2.imshow(str(i), output[i])

cv2.waitKey(0)
cv2.destroyAllWindows()