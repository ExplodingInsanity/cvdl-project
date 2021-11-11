import cv2

def resize(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim)

img = cv2.imread('./unknown.png')
tmp = cv2.imread('./images/left_center.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
tmp = resize(tmp , 50)
w = tmp.shape[0]
h = tmp.shape[1]
print(img.shape)
print(tmp.shape)
# print(tmp.shape[::1])
result = cv2.matchTemplate(img , tmp, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
x,y = max_loc
x1 ,y1 = x + h , y + w
cv2.rectangle(img , (x,y) , (x1,y1) , 255, 2)
cv2.putText(img, str(max_val), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
while True :
    cv2.imshow('window',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break