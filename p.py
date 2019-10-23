

		
def maxSum(arr, n, k):
    res = 0
    start=0
    for i in range(k):
        res += arr[i]
    curr_sum = res
    for i in range(k, n):
        curr_sum += arr[i] - arr[i-k]
        if res < curr_sum:
            res=curr_sum
            start=i-k+1
    if res>0:
        return start
    else:
        return -1

def crop(a):
    t=a.shape
    count=0
    f=0
    for i in range(0,t[0]):
        for j in range(0,t[1]):
            if a[i][j]!=0:
                count+=1
                t1=i
                f=1
                break
        if f==1:
            break
        
    if count==0:
        return 0
    f=0
    for i in range(0,t[1]):
        for j in range(0,t[0]):
            if a[j][i]!=0:
                t2=i
                f=1
                break
        if f==1:
            break
    f=0
    for i in range(t[0]-1,-1,-1):
        for j in range(0,t[1]):
            if a[i][j]!=0:
                t3=i
                f=1
                break
        if f==1:
            break
    f=0
    for i in range(t[1]-1,-1,-1):
        for j in range(0,t[0]):
            if a[j][i]!=0:
                t4=i
                f=1
                break
        if f==1:
            break
    if t1>0:
        t1-=1
    if t2>0:
        t2-=1
    if t3<t[0]-1:
        t3+=1
    if t4<t[1]-1:
        t4+=1

    rows=t3-t1+1
    colm=t4-t2+1
    a1=np.zeros((rows,colm), dtype=np.uint8)
    r=0
    c=0
    for i in range(t1,t3+1):
        c=0
        for j in range(t2,t4+1):
            a1[r][c]=a[i][j]
            c+=1
        r+=1
    return a1

def cnn(s1):
    b=crop(s1)
    cv2.imshow("crop",b)
    newImage = cv2.resize(b,(28, 28))
    newImage = np.array(newImage)
    newImage = newImage.astype('float32')/255
    prediction2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
    prediction2 = np.argmax(prediction2)
    return str(letters[int(prediction2)+1])


from keras.models import load_model
import numpy as np
import copy
import cv2

cnn_model = load_model('emnist_cnn_model.h5')

letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

cap = cv2.VideoCapture(0)
count1=0
l=[]
count=0
while(True):
    
    if count%2==0:
        count+=1
        continue
    # Capture frame-by-frame
    ret, frame = cap.read()
    if len(l)>0:
        for i in range(0,len(l)):
            if i==0:
                continue
            cv2.line(frame , l[i-1] , l[i] , (0,0,255) , 3)
    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #blue hsv range
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    kernel1 = np.ones((5,5), np.uint8)
    erosion = cv2.erode(mask,kernel1,iterations = 2)
    dilation = cv2.dilate(erosion,kernel1,iterations = 2)
    ret,thresh1 = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY)    
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    
    t=thresh1.shape
    alpha=np.zeros((t[0],t[1]), dtype=np.uint8)
    horizontal_hist=[0]*t[0]
    vertical_hist=[0]*t[1]

    
    for i in range(0,t[0]):
        for j in range(0,t[1]):
            if thresh1[i][j]==255:
                vertical_hist[j]+=1
                horizontal_hist[i]+=1
    k=50
    col_start=maxSum(vertical_hist,t[1],k)
    row_start=maxSum(horizontal_hist,t[0],k)

    if col_start + row_start >0:
        count1=0
        if col_start+k > t[1]-1:
            col_end = t[1]
        else:
            col_end = col_start+k

        if row_start+k > t[0]-1:
            row_end = t[0]
        else:
            row_end = row_start+k

        rect = cv2.rectangle(frame, (col_start,row_start), (col_end,row_end), (0, 0, 255), 2)
        x=int((col_start + col_end)/2)
        y=int((row_start + row_end)/2)
        rect1 = cv2.rectangle(rect, (x,y), (x+2,y+2), (0, 0, 255), 2)
        
        l.append((x,y))
        
    # Display the resulting frame
        cv2.imshow('frame',rect1)
    else:
        count1+=1
        if count1>3:
            count1=0
            if len(l)>0:
                #print(l)
                for i in range(0,len(l)):
                    if i==0:
                        continue
                    cv2.line(alpha , l[i-1] , l[i] , 255 , 3)
                    
                alpha1 = cv2.flip(alpha, 1)
                cv2.imshow("alpha",alpha1)
                print(str(cnn(alpha1)))
                
            l=[]
        cv2.imshow('frame',frame)
    count+=1    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
