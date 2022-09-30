import cv2
import sys
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

size = 30
cnt = 20
out_dir = 'output'

def crop_texture(dst):
    '''
    crop large size texture image
    '''
    imgs = []
    for i in range(0, size*cnt, size): # 600
        for j in range(0, size*cnt, size):
            crop = dst[i:i+size, j:j+size]
            imgs.append(crop)
    return imgs

def image_operation(digits_imgs, mask):
    '''
    Add damaged mask and gray background
    return image array list
    '''
    # 배경 생성
    bg = np.random.randint(30, size=(size, size), dtype=np.uint8)
    bg = bg + 150 # color range 150 ~ 180

    # 블러 커널, 밝기 낮추는 커널
    kernel = np.ones((3,3), np.float32) / 9
    base = np.full((size, size), 170, dtype=np.uint8)

    output = []
    for i in range(len(digits_imgs)):
        # Add mask on digits image
        damaged = cv2.add(digits_imgs[i], mask)
        # 숫자 부분만 밝히기 위해 배경 배경은 0, 숫자 부분은 n값을 가지는 마스크 만듬
        bright_mask = cv2.subtract(damaged, base) # pixel range 0 ~ 85 (255 - 170)
        gray_bg = cv2.subtract(bg, bright_mask) # add background 
        blur = cv2.filter2D(gray_bg, -1, kernel) # blur
        output.append(blur)
    return output

def make_dirs(output_path):
    '''
    make output dir
    '''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        for i in range(10):
            sub_path = os.path.join(output_path, str(i))
            os.mkdir(sub_path)
    else:
        for i in range(10):
            sub_path = os.path.join(output_path, str(i))
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
            

# 텍스처 이미지 로드
texture = cv2.imread('img/crack_texture.png', 0)
_, t_dst = cv2.threshold(texture, 127, 255, cv2.THRESH_BINARY)
t_dst = 255 - t_dst

# crop
w, h = texture.shape
if w > size*cnt and h > size*cnt:
    t_dst = t_dst[0:size*cnt , 0:size*cnt] # size (600, 600)
else:
    sys.exit()

# crop image (30, 30)
t_mask = crop_texture(t_dst)
print(f'Num of texture mask {len(t_mask)}')

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
digits_imgs = []
for i in range(len(digits_coordinate)):
    x, y, w, h = digits_coordinate[i]
    # crop
    mask = dst[y:y+h, x:x+w]

    # add padding
    h_pad = (size - h) >> 1
    w_pad = (size - w) >> 1

    # add black border 
    border = cv2.copyMakeBorder(mask, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, None, 0)
    resized = cv2.resize(border, (size, size))

    digits_imgs.append(resized)

# Add damaged and gray mask
# test = image_operation(digits_imgs, t_mask[10])

# write image
make_dirs(out_dir) # output/0 ~ 9
cnt = 0
for i in range(len(t_mask)):
    output_list = image_operation(digits_imgs, t_mask[i])
    for i in range(len(output_list)):
        f_dir = os.path.join(out_dir, str(i))
        f_name = 'digits_'+str(i)+'_'+str(cnt)+'.png'
        cv2.imwrite(os.path.join(f_dir, f_name), output_list[i])
        print(f'Write image {f_name}')
        cnt = cnt + 1