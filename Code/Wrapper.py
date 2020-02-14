

import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
import os

def gauss1d(sigma,mean,x,oder):
    x=x-mean
    x_=np.square(x)
    g=np.exp(-1*x_/(2*(sigma**2)))
    g=g/np.sqrt(2*np.pi*(sigma**2))
    if (oder==0):
        return g
    elif (oder==1):
        g=-1*g*(x/(sigma**2))
        return g
    elif (oder==2):
        g=g*((x_-(sigma*2))/(sigma*4))
        return g

def makefilter(scale,x,y,pts,dim):
    g_x=gauss1d(3*scale,0,pts[0],x)
    g_y=gauss1d(scale,0,pts[1],y)
    #print("g_y = ",type(g_y))
    filt=g_x*g_y
    #print("filt = ",filt.shape," dim = ",dim)
    filt = np.reshape(filt,(dim,dim))
    return filt

def makefilterDOG(scale,x,y,pts,dim):
    g_x=gauss1d(scale,0,pts[0],x)
    g_y=gauss1d(scale,0,pts[1],y)
    #print("g_y = ",type(g_y))
    filt=g_x*g_y
    #print("filt = ",filt.shape," dim = ",dim)
    filt = np.reshape(filt,(dim,dim))
    return filt

def gaussian2D(dim, sd):
    var = sd**2
    x_org = (dim - 1)/2
    y_org = (dim - 1)/2
    x,y = np.ogrid[-x_org:x_org+1,-y_org:y_org+1]
    G = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return G

def LOG2d(dim, sd):
    var = sd**2
    x_org = (dim - 1)/2
    y_org = (dim - 1)/2
    x,y = np.ogrid[-x_org:x_org+1,-y_org:y_org+1]
    temp = np.exp( -(x*x + y*y) / (2*var) )*(1-(x*x + y*y) / (2*var))
    G = (-1/(np.pi*(var**2)))*temp
    #G = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return G

def makeDOGfilter(dim,scale,norient):
    scale_x=np.sqrt(2) ** np.arange(1,scale+1)
    total_filters = scale*norient
    F = np.zeros([dim,dim,total_filters])
    temp_org = (dim - 1)/2
    x = [np.arange(-temp_org,temp_org+1)]
    y = [np.arange(-temp_org,temp_org+1)]
    [x,y] = np.meshgrid(x,y)
    all_pts = np.array([x.flatten(), y.flatten()])
    count=0
    for scale in range(len(scale_x)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotated_pts = np.array([[c,-s],[s,c]])
            rotated_pts = np.dot(rotated_pts,all_pts)
            F[:,:,count] = makefilterDOG(scale_x[scale], 0, 1, rotated_pts, dim)
            count = count + 1
    return F

def makeLMSfilters(dim,scale,norient, nrotinv):
    scale_x=np.sqrt(2) ** np.arange(0,scale)
    first_order  = len(scale_x)*norient
    second_order = len(scale_x)*norient
    total_filters = first_order + second_order + nrotinv
    F = np.zeros([dim,dim,total_filters])
    temp_org = (dim - 1)/2
    x = [np.arange(-temp_org,temp_org+1)]
    y = [np.arange(-temp_org,temp_org+1)]
    [x,y] = np.meshgrid(x,y)
    all_pts = np.array([x.flatten(), y.flatten()])
    count = 0
    for scale in range(len(scale_x)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotated_pts = np.array([[c,-s],[s,c]])
            rotated_pts = np.dot(rotated_pts,all_pts)
            F[:,:,count] = makefilter(scale_x[scale], 0, 1, rotated_pts, dim)
            F[:,:,count+second_order] = makefilter(scale_x[scale], 0, 2, rotated_pts, dim)
            count = count + 1
    count = first_order +second_order
    sd = np.sqrt(2) ** np.array([0,1,2,3])
    for i in range(len(sd)):
        F[:,:,count]   = gaussian2D(dim, sd[i])
        count = count + 1
    for i in range(len(sd)):
        F[:,:,count]   = LOG2d(dim, sd[i])
        count = count + 1
    for i in range(len(sd)):
        F[:,:,count]   = LOG2d(dim, 3*sd[i])
        count = count + 1
    return F

def makeLMLfilters(dim,scale,norient, nrotinv):
    scale_x=np.sqrt(2) ** np.arange(1,scale+1)
    first_order  = len(scale_x)*norient
    second_order = len(scale_x)*norient
    total_filters = first_order + second_order + nrotinv
    F = np.zeros([dim,dim,total_filters])
    temp_org = (dim - 1)/2
    x = [np.arange(-temp_org,temp_org+1)]
    y = [np.arange(-temp_org,temp_org+1)]
    [x,y] = np.meshgrid(x,y)
    all_pts = np.array([x.flatten(), y.flatten()])
    count = 0
    for scale in range(len(scale_x)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotated_pts = np.array([[c,-s],[s,c]])
            rotated_pts = np.dot(rotated_pts,all_pts)
            F[:,:,count] = makefilter(scale_x[scale], 0, 1, rotated_pts, dim)
            F[:,:,count+second_order] = makefilter(scale_x[scale], 0, 2, rotated_pts, dim)
            count = count + 1
    count = first_order +second_order
    sd = np.sqrt(2) ** np.array([0,1,2,3])
    for i in range(len(sd)):
        F[:,:,count]   = gaussian2D(dim, sd[i])
        count = count + 1
    for i in range(len(sd)):
        F[:,:,count]   = LOG2d(dim, sd[i])
        count = count + 1
    for i in range(len(sd)):
        F[:,:,count]   = LOG2d(dim, 3*sd[i])
        count = count + 1
    return F

def makeGABORfilters(dim,scale,norient,lamda,gamma,phi):
    temp_org = (dim - 1)/2
    x = [np.arange(-temp_org,temp_org+1)]
    y = [np.arange(-temp_org,temp_org+1)]
    [x,y] = np.meshgrid(x,y)
    all_pts = np.array([x.flatten(), y.flatten()])
    #scale_x=np.sqrt(2) ** np.arange(1,scale+1)
    scale_x=2.3 ** np.arange(1,scale+1)
    total_filters = scale * norient
    F = np.zeros([dim,dim,total_filters])
    count=0
    for i in range(len(scale_x)):
        for j in range(norient):
            angle = (np.pi * j)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotated_pts = np.array([[c,-s],[s,c]])
            rotated_pts = np.dot(rotated_pts,all_pts)
            x_theta=rotated_pts[0]
            y_theta=rotated_pts[1]
            g = np.exp(-0.5 * (x_theta ** 2 / scale_x[i] ** 2 + (y_theta ** 2) *(gamma**2) / scale_x[i] ** 2)) * np.cos(2 * np.pi / lamda * x_theta + phi)
            F[:,:,count] = np.reshape(g,(dim,dim))
            count=count+1
    return F

def makeTextonMap(img,filters):
    txt = np.zeros((img.shape[0],img.shape[1],filters.shape[2]),dtype='uint8')
    count=0
    for i in range(filters.shape[2]):
        dst = cv2.filter2D(img,-1,filters[:,:,i])
        txt[:,:,count]=dst
        count=count+1
    return txt

def plot_DOG(filters):
    _,_,size = filters.shape
    plt.subplots(2,16,figsize=(160,20))
    for i in range(size):
        plt.subplot(2,16,i+1)
        plt.axis('off')
        plt.imshow(filters[:,:,i],cmap='gray')
    plt.savefig('DOGfilterbank.png')
    plt.close()

def plot_LM(filters):
    _,_,size = filters.shape
    plt.subplots(4,12,figsize=(60,20))
    for i in range(size):
        plt.subplot(4,12,i+1)
        plt.axis('off')
        plt.imshow(filters[:,:,i],cmap='gray')
    plt.savefig('LMfilterbank.png')
    plt.close()

def plot_Gabor(filters):
    _,_,size = filters.shape
    plt.subplots(5,8,figsize=(30,20))
    for i in range(size):
        plt.subplot(5,8,i+1)
        plt.axis('off')
        plt.imshow(filters[:,:,i],cmap='gray')
    plt.savefig('Gaborfilterbank.png')
    plt.close()    


def createHalfDisc():
    scales = [10, 26, 40]
    orientation = [0, 22.5,45,67.5, 90,112.5,135,157.5]
    discList = []
    for scale in scales:
        for orient in orientation:
            discImg1 = np.zeros((scale,scale))
            discImg2 = np.zeros((scale,scale))
            axes = (int((scale)/2), int((scale)/2))
            startAngle = orient;
            endAngle = orient + 180
            center = (int((scale/2)-1), int((scale/2)-1))
            cv2.ellipse(discImg1, center, axes, 0, startAngle, endAngle, 255, -1)
            cv2.ellipse(discImg2, center, axes, 0, startAngle + 180, endAngle + 180, 255, -1)
            discList.append(discImg1)
            discList.append(discImg2)
    return discList

def gradient(maps,bins, filter_bank):
    chi_sqr_dist = np.zeros((maps.shape[0],maps.shape[1]))
    g = np.zeros((maps.shape[0],maps.shape[1]))
    h = np.zeros((maps.shape[0],maps.shape[1]))
    gradVar = maps
    for N in range(len(filter_bank)/2):
        chi_sqr_dist = chi_sqr_dist*0
        g = g*0
        h = h*0
        for i in range(bins):
            tmpimg = np.ma.masked_where(maps == i,maps)
            tmpimg = tmpimg.mask.astype(np.int)
            g = cv2.filter2D(tmpimg,-1,filter_bank[2*N])
            h = cv2.filter2D(tmpimg,-1,filter_bank[2*N+1])
            chi_sqr_dist = chi_sqr_dist + ((g-h)**2 /(g+h))
        chi_sqr_dist = chi_sqr_dist/2
        #g = chi_sqr_gradient(maps, bins, filter_bank[2*N],filter_bank[2*N+1])
        gradVar = np.dstack((gradVar,chi_sqr_dist))
    mean = np.mean(gradVar,axis =2)
    return mean
    

# def GaborFilterCV2(): #(ksize, sigma, theta, lambda, gamma, psi, ktype)
#     filters = []
#     ksize = 51
#     for theta in np.arange(0, np.pi, np.pi / 16):
#         kern = cv2.getGaborKernel((ksize, ksize), 8, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#         kern /= 1.5*kern.sum()
#         filters.append(kern)
#     return filters

DOGFilter = makeDOGfilter(13,2,16)
print("DOGFilter.shape = ",DOGFilter.shape)
plot_DOG(DOGFilter)
LMLfilter = makeLMLfilters(49,3,6, 12)
LMSfilter = makeLMSfilters(49,3,6, 12)
print("LMfilter.shape = ",LMLfilter.shape)
plot_LM(LMLfilter)
GaborFilter = makeGABORfilters(31,5,8,0.523,0.5,0)
print("GaborFilter.shape = ",GaborFilter.shape)
plot_Gabor(GaborFilter)

total_filters = DOGFilter.shape[2] + LMLfilter.shape[2] + LMSfilter.shape[2] + GaborFilter.shape[2]

count=1
imagesI = []
imagesC = []
imagesS = []
pathI = "/home/raghav/Desktop/YourDirectoryID_hw0/Phase1/BSDS500/Images/"
pathC = "/home/raghav/Desktop/YourDirectoryID_hw0/Phase1/BSDS500/CannyBaseline/"
pathS = "/home/raghav/Desktop/YourDirectoryID_hw0/Phase1/BSDS500/SobelBaseline/"

for img in os.listdir(pathI):
    imagesI.append(img)
for img in os.listdir(pathC):
    imagesC.append(img)
for img in os.listdir(pathS):
    imagesS.append(img)
#print(images.shape)

imagesI.sort()
imagesC.sort()
imagesS.sort()

for i in range(len(imagesI)):
    image = cv2.imread("%s%s" % (pathI, imagesI[i]))
    print(image.shape)

    # image = cv2.imread("C:\\Users\\ragha\\Desktop\\116321549_hw0\\Phase1\\BSDS500\\Images\\1.jpg")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.filter2D(img,-1,GaborFilter[:,:,3])
    textonDOG = makeTextonMap(img,DOGFilter)
    textonLML = makeTextonMap(img,LMLfilter)
    textonLMS = makeTextonMap(img,LMSfilter)
    textonGabor = makeTextonMap(img,GaborFilter)
    print("textonDOG = ",textonDOG.shape," textonLML = ",textonLML.shape," textonLMS = ",textonLMS.shape, " textonGabor = ",textonGabor.shape)
    all_texton = np.dstack((textonDOG,textonLML,textonLMS,textonGabor))
    print(all_texton.shape)
    all_texton_r = np.reshape(all_texton,(img.shape[0]*img.shape[1],total_filters))
    kmeanstxt = KMeans(n_clusters=64, random_state=2).fit(all_texton_r)
    TextonMap=kmeanstxt.predict(all_texton_r)

    TextonMap = np.reshape(TextonMap,(img.shape[0],img.shape[1]))
    print(TextonMap.shape)
    plt.subplot(131)
    plt.imshow(TextonMap)


    brightHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_map = np.reshape(brightHSV[:,:,2],(img.shape[0]*img.shape[1],1))
    kmeansbri = KMeans(n_clusters=16, random_state=2).fit(brightness_map)
    BrightnessMap=kmeansbri.predict(brightness_map)
    BrightnessMap = np.reshape(BrightnessMap,(img.shape[0],img.shape[1]))
    plt.subplot(132)
    plt.imshow(BrightnessMap)
    # plt.show()
    # cv2.imshow('image',brightness_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(brightness_map.shape)


    color_map = np.reshape(image,(img.shape[0]*img.shape[1],3))
    kmeanscol = KMeans(n_clusters=16, random_state=2).fit(color_map)
    ColorMap=kmeanscol.predict(color_map)
    ColorMap = np.reshape(ColorMap,(img.shape[0],img.shape[1]))
    plt.subplot(133)
    plt.imshow(ColorMap)
    stri = "%s%s.png" % ("Maps", count) 
    #plt.show()
    plt.savefig(stri, dpi=350)
    plt.close()
    count=count+1
    print("loop iteration ",count-1," completed")

    #plt.show()
    HalfDiscs = createHalfDisc()
    Texton_Gradient = gradient(TextonMap,64, HalfDiscs)
    plt.subplot(131)
    plt.imshow(Texton_Gradient)
    Brightness_Gradient = gradient(BrightnessMap,16, HalfDiscs)
    plt.subplot(132)
    plt.imshow(Brightness_Gradient)
    Color_Gradient = gradient(ColorMap,16, HalfDiscs)
    plt.subplot(133)
    plt.imshow(Color_Gradient)
    stri = "%s%s.png" % ("Gradient", count) 
    #plt.show()
    plt.savefig(stri, dpi=350)
    plt.close()

    Temp_Gradient = (Texton_Gradient + Brightness_Gradient + Color_Gradient)/3
    sobel = cv2.imread("%s%s" % (pathS, imagesS[i]),0)
    canny = cv2.imread("%s%s" % (pathC, imagesC[i]),0)

    w1=0.5
    w2=0.5

    PBlite = np.multiply(Temp_Gradient,(w1*canny + w2*sobel))
    stri = "%s%s.jpg" % ("PBOutput", count)

    plt.imshow(PBlite,cmap='gray')
    plt.savefig(stri)
    plt.close()
    #plt.show()
    #cv2.imwrite(stri, PBlite)
    #cv2.imshow('image',PBlite)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
