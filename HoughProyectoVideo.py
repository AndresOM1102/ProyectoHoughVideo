import cv2
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
global t1
import matplotlib.pyplot as plt

def playVideo():
    file = input("Enter the path to the MP4 video file: ")
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("Error")
        exit(0)

    print("Propiedades")
    ancho=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    alto=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Dimensiones: ",ancho,alto)
    print("FPS: ", cap.get(cv2.CAP_PROP_FPS))

    while( cap.isOpened()):
        ret, img = cap.read()
        if ret:
            imgin = img[5,5,(2,1,0)]#OPEN CV esta en el orden BGR
        else:
            break
        imgPA=HoughTransform(img)
        plt.imshow(imgPA)
        plt.show()
        time.sleep(1/cap.get(cv2.CAP_PROP_FPS))
    cap.release()


def Prewitt_Conv(imagen):
    Prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_gaussian = cv2.GaussianBlur(imagen,(3,3),0)
    img_prewittx = cv2.filter2D(img_gaussian, -1, Prewitt)
    img_prewitty = cv2.filter2D(img_gaussian, -1, Prewitt.T)
    img_prwt= img_prewittx + img_prewitty

    kernel = np.ones((5, 5), np.uint8)
    binary_img = cv2.morphologyEx(img_prwt, cv2.MORPH_CLOSE, kernel)
    return (binary_img)


def Filtro(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_conv=Prewitt_Conv(gray)

    th=0.60*255
    for i in range(img_conv.shape[0]):
        for j in range(img_conv.shape[1]):
            if (img_conv[i][j]>th):
                img_conv[i][j]=255
            else:
                img_conv[i][j]=0

    return img_conv

def Hough(frame):
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = Filtro(frame)
    t1 = time.time()

    filas=frame.shape[0]
    columnas=frame.shape[1]

    diag=int(np.round(np.sqrt(columnas**2+filas**2),0))
    acumulado=np.zeros(diag*2*180).reshape(diag*2,180)
    th=0.85
    i=0
    for x in range(filas):
        for y in range(columnas):
           if (frame[x][y] > th):
                for ang in range(180):
                    p=int(y*np.cos(np.deg2rad(ang))+(filas-1-x)*np.sin(np.deg2rad(ang)))
                    acumulado[p+diag-1][ang]+=1
                    i+=1
    return acumulado, t1


def y(a,b,r,x):
    y_val = (r-a*x)/b
    return y_val

def x(a,b,r,y):
    x_val = (r-y*b)/a
    return x_val


def distance (x0,x1,x2,x3,dt):
    d = np.sqrt( (x3-x1)**2 +  (x2-x0)**2)
    t = dt/d
    xd = ((1-t)*x0 + t*x2)
    yd = ((1-t)*x1 + t*x3)
    return (xd,yd)

def polar2space(a,b,r,alto,ancho):
    alto -= 1
    ancho -= 1
    points = []

    # para limitar tamaño de lineas
    d = np.sqrt(alto**2 + ancho**2)/4 # tamaño de la linea

    if ( abs(a) == 0):
        horizontal = r/b
        return [0,alto-horizontal,ancho,alto-horizontal]
    if ( abs(b) == 0):
        vertical = r/a
        return [vertical, alto, vertical, alto -d]

    y_x0 = y(a,b,r,0)
    y_xmax = y(a,b,r,ancho)
    x_y0 = x(a,b,r,0)
    x_ymax = x(a,b,r,alto)


    if ( x_y0 >= 0 and x_y0<= ancho):
        points.append(x_y0)
        points.append(alto)

    if ( x_ymax >= 0 and x_ymax<= ancho):
        points.append(x_ymax)
        points.append(0)

    if (y_x0 >= 0 and y_x0 <= alto):
        points.append(0)
        points.append(alto-y_x0)

    if (y_xmax >= 0 and y_xmax <= alto):
        points.append(ancho)
        points.append(alto-y_xmax)

    if (points[1]<points[3]):

        punto = distance(points[2],alto-points[3],points[0],alto-points[1],d)
        points[0] = punto[0]
        points[1] = alto-punto[1]
        return points

    elif (points[3]<points[1]):

        punto = distance(points[0],alto-points[1],points[2],alto-points[3],d)
        points[2] = punto[0]
        points[3] = alto-punto[1]

    return points


def HoughTransform(frame):
    t0=time.time()
    acumulado,t1=Hough(frame)
    imagen=frame
    imagen=cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
    alto = imagen.shape[0]
    ancho = imagen.shape[1]
    diag = int(np.round(np.sqrt(alto ** 2 + ancho ** 2), 0))

    max_value = max([max(l) for l in acumulado])
    acumulado = acumulado * 255 / (max_value)
    matplotlib.image.imsave('espacio_h.png', acumulado) #Guardamos Hough space como imagen
    espacio_h = cv2.imread('espacio_h.png') #Abrimos la imagen
    espacio_h = cv2.cvtColor(espacio_h, cv2.COLOR_BGR2RGB)
    espacio_h = cv2.resize(espacio_h,(500,1000))

    t2=time.time()

    umbral = 80
    maximos = np.where(acumulado > umbral)
    indices_f = maximos[0]
    indices_c = maximos[1]

    for i in range(len(indices_f)):
        theta = indices_c[i]
        if not(theta>=80 and theta<=100):
          rho = indices_f[i] + 1 - diag
          a= np.cos(np.deg2rad(theta)) # cos theta
          b = np.sin(np.deg2rad(theta)) # sin theta
          puntos = polar2space(a,b,rho,alto,ancho)
          pt1 = ( int(puntos[0]), int(puntos[1]))
          pt2 = (int(puntos[2]), int(puntos[3]))
          cv2.line(imagen, pt1, pt2, (255, 0, 0), 2)
    t3=time.time()

    print("Tiempo filtro: ",t1-t0)
    print("Tiempo Hough: ",t2 - t1)
    print("Tiempo ploteo: ",t3 - t2)
    return imagen

playVideo()