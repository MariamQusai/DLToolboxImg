from imports import *
class ctscan():
    def __init__(self,scan_id):
        self.qu = pl.query(pl.Scan) #load data

        self.scan_id = scan_id

        self.scans = self.qu.all() 

        self.scan=self.scans[self.scan_id-1]

        self.slices= self.scan.load_all_dicom_images() #all raw slices of ctscan with metadata

        self.flag=0
    
    

        if len(self.slices)>1:
            for s in self.slices:
                try:
                  s.pixel_array
                except:
                  self.flag=-1
    
        if (self.flag==0 and len(self.slices)>1):
            self.image= np.stack([s.pixel_array for s in self.slices])
            self.image_HU=self.get_pixels_hu() #all slices of ctscan in Housenfield unit

            self.slice_thickness=self.slices[1].ImagePositionPatient[2]-self.slices[0].ImagePositionPatient[2]
            self.original_spacing = [self.scan.pixel_spacing,self.scan.pixel_spacing,
                                   self.slice_thickness]    
            self.desired_spacing = np.array([1,1,1])

            self.image_resampled,self.new_spacing=self.resample()
            self.image_normalized = self.normalize(self.image_resampled)

            self.z0=float(self.slices[0].ImagePositionPatient[2]) #they are sorted by pylidc #get initial z position

            self.anns= self.scan.annotations

            self.centroids1=self.allcentroids1()

            self.centroids2=self.allcentroids2()

            self.zarrs = [self.get_ann_z(ann) for ann in self.anns] #for each annotations get the z axis that it spans

            self.c2vsz=self.find_all_centroids()
            
            self.Zbbox=self.find_all_bbox()

            self.Z2=self.find_msk()


    def get_pixels_hu(self):
        image = self.image.astype(np.float32)
        image[image<-1000]=0
        for slice_number in range(len(self.slices)):
            intercept = self.slices[slice_number].RescaleIntercept
            slope = self.slices[slice_number].RescaleSlope
            image[slice_number] = slope * image[slice_number]
            image[slice_number] += intercept
        return np.array(image, dtype=np.int16) #zxy
  
    def resample(self):

        image=self.image_HU.copy() #zxy

        image=image.swapaxes(0,2) #yxz
        image=image.swapaxes(0,1) #xyz

        resize_factor = self.original_spacing/self.desired_spacing

        new_real_shape = image.shape * resize_factor

        new_shape = np.round(new_real_shape)

        real_resize_factor = new_shape/image.shape

        new_spacing = self.original_spacing/real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        image=image.swapaxes(0,2) #zyx
        image=image.swapaxes(1,2) #zxy

        return image, new_spacing


    def normalize(self,image):
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0

        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image

    def centroid1(self,ann): #convert centroid from mm into pixel location#centroid in coordinates1
        #find centroid
        row,col,depth =ann.centroid #row,col,z in original dimensions,m
        c=np.array([col,row,depth])
        #find centroid with z in slice thickness
        c1=c.copy().astype(int)
        #c1[2]=np.int((c[2]-self.z0)/self.slice_thickness)
        return c1

    def centroid2(self,c1): #convert into spacing of 1,1,1
        c2=c1*self.original_spacing
        c2=c2.astype(np.int)
        return c2

    def allcentroids1(self): #all nodule centroids in the original coordinate system
        centroids=[]
        for ann in self.anns:
            c1=self.centroid1(ann)
            centroids.append(c1)
        return centroids
    
    def allcentroids2(self): #all nodule centroids in the resampled coordinate system
        centroids=[]
        for c1 in self.centroids1:
            c2=self.centroid2(c1)
            centroids.append(c2)
        return centroids
    
    def bbox1(self,ann): #convert bbox from mm into pixel location
        #A=ann.bbox()
        #A[2]=(A[2]*1.0-self.z0)/self.slice_thickness
        A=ann.bbox_matrix() #yxz
        A=np.array([A[1],A[0],A[2]])
        return A
    
    def bbox2(self,ann): #convert into spacing of 1,1,1
        A1=self.bbox1(ann)
        A2=A1.T*self.original_spacing
        A2=A2.T
        return A2
    def get_ann_z(self,ann): #find the z that a nodule spans
        zs=[]
        A2=self.bbox2(ann)
        zarr=np.arange(A2[2][0],A2[2][1]+1).astype(np.int)
        return zarr

    #find centroid at each z location of the given volume
    def find_all_centroids(self):
        slices_num = self.image_resampled.shape[0]
        #initialize dict
        keys=np.arange(slices_num)
        Zc=dict() #The dict contains a mask for each z location
        [Zc.setdefault(key, []) for key in keys] #contains a centroid for each z

        for i,c2 in enumerate(self.centroids2):
            for key in self.zarrs[i]:
                Zc[key].append(c2[0:2])
        return Zc
    #find bottem left corner and dimension of bbox at each z location of the given volume
    def find_all_bbox(self):
        slices_num = self.image_resampled.shape[0]
        
        keys=np.arange(slices_num)
        Zxy=dict() #The dict contains a mask for each z location
        [Zxy.setdefault(key, []) for key in keys] #contains a centroid for each z
        
        for i,(ann,c2) in enumerate(zip(self.anns,self.centroids2)):
            A2=self.bbox2(ann)
            d1=A2.T[1]-A2.T[0]#bbox dim
            xy1=[c2[0]-d1[0]/2.,c2[1]-d1[1]/2.]
            for key in self.zarrs[i]:
                Zxy[key].append(xy1[0:2]+list(d1[0:2]))
        return Zxy
    def find_msk(self):
        msk = 0*self.image_resampled
        for ann in self.anns:
            x1,x2=np.floor(self.bbox2(ann)[0]).astype(np.int)
            y1,y2=np.floor(self.bbox2(ann)[1]).astype(np.int)
            z1,z2=np.floor(self.bbox2(ann)[2]).astype(np.int)
            msk[z1:z2+1,y1:y2+1,x1:x2+1]=1
        return msk

def get_cube_from_img(img3d, center_x, center_y, center_z, block_size): #image zyx?

    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res,[start_x,start_y,start_z]
