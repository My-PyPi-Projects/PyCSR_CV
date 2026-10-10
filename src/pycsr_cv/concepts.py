"""Learning catalog and executable operations for classic computer vision."""

from dataclasses import dataclass
from typing import Callable, Dict, List

import cv2
import numpy as np


@dataclass(frozen=True)
class Concept:
    name: str
    category: str
    meaning: str
    effect: str
    operation: Callable[[np.ndarray], np.ndarray]
    params: str = ""


def gray(i): return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
def binary(i): return cv2.threshold(gray(i),127,255,cv2.THRESH_BINARY)[1]
def adaptive(i): return cv2.adaptiveThreshold(gray(i),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,5)
def otsu(i): return cv2.threshold(gray(i),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
def gaussian(i): return cv2.GaussianBlur(i,(9,9),0)
def median(i): return cv2.medianBlur(i,9)
def bilateral(i): return cv2.bilateralFilter(i,9,75,75)
def sharpen(i): return cv2.filter2D(i,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32))
def laplacian(i): return cv2.convertScaleAbs(cv2.Laplacian(gray(i),cv2.CV_64F))
def canny(i): return cv2.Canny(gray(i),100,200)
def sobelx(i): return cv2.convertScaleAbs(cv2.Sobel(gray(i),cv2.CV_64F,1,0,ksize=3))
def sobely(i): return cv2.convertScaleAbs(cv2.Sobel(gray(i),cv2.CV_64F,0,1,ksize=3))
def scharr(i): return cv2.convertScaleAbs(cv2.Scharr(gray(i),cv2.CV_64F,1,0))
def hsv(i): return cv2.cvtColor(i,cv2.COLOR_BGR2HSV)
def lab(i): return cv2.cvtColor(i,cv2.COLOR_BGR2LAB)
def ycrcb(i): return cv2.cvtColor(i,cv2.COLOR_BGR2YCrCb)
def hls(i): return cv2.cvtColor(i,cv2.COLOR_BGR2HLS)
def equalize(i): return cv2.equalizeHist(gray(i))
def clahe(i): return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray(i))
def invert(i): return cv2.bitwise_not(i)
def gamma(i): return np.uint8(np.clip((i/255.0)**0.6*255,0,255))
def log_transform(i): return np.uint8(np.clip(255/np.log1p(255)*np.log1p(i.astype(np.float32)),0,255))
def erode(i): return cv2.erode(binary(i),np.ones((5,5),np.uint8),iterations=1)
def dilate(i): return cv2.dilate(binary(i),np.ones((5,5),np.uint8),iterations=1)
def opening(i): return cv2.morphologyEx(binary(i),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
def closing(i): return cv2.morphologyEx(binary(i),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
def gradient(i): return cv2.morphologyEx(binary(i),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
def tophat(i): return cv2.morphologyEx(gray(i),cv2.MORPH_TOPHAT,np.ones((15,15),np.uint8))
def blackhat(i): return cv2.morphologyEx(gray(i),cv2.MORPH_BLACKHAT,np.ones((15,15),np.uint8))
def contours(i):
    out=i.copy(); cs,_=cv2.findContours(binary(i),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out,cs,-1,(40,230,130),2); return out
def harris(i):
    out=i.copy(); c=cv2.cornerHarris(np.float32(gray(i)),2,3,0.04); out[c>0.01*c.max()]=(0,0,255); return out
def good_features(i):
    out=i.copy(); pts=cv2.goodFeaturesToTrack(gray(i),200,0.01,10)
    if pts is not None:
        for p in np.intp(pts).reshape(-1,2): cv2.circle(out,tuple(p),4,(0,255,255),-1)
    return out
def orb(i):
    out=i.copy(); k=cv2.ORB_create(nfeatures=400).detect(gray(i),None); return cv2.drawKeypoints(out,k,None,color=(0,255,255),flags=0)
def sift(i):
    out=i.copy()
    if not hasattr(cv2,"SIFT_create"): raise ValueError("SIFT is unavailable in this OpenCV build.")
    k=cv2.SIFT_create(nfeatures=250).detect(gray(i),None); return cv2.drawKeypoints(out,k,None,color=(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
def hog(i):
    h=cv2.HOGDescriptor((128,128),(16,16),(8,8),(8,8),9); v=h.compute(cv2.resize(i,(128,128))).reshape(-1); canvas=np.zeros((220,420,3),np.uint8)
    for n,x in enumerate(v[:180]): cv2.line(canvas,(10+n*2,205),(10+n*2,205-min(170,int(x*500))),(90,210,255),1)
    cv2.putText(canvas,"HOG: local gradient-orientation histogram",(10,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1); return canvas
def fourier(i):
    g=gray(i); f=np.fft.fftshift(np.fft.fft2(g)); mag=cv2.normalize(20*np.log(np.abs(f)+1),None,0,255,cv2.NORM_MINMAX); return mag.astype(np.uint8)
def dct(i):
    g=cv2.resize(gray(i),(256,256)).astype(np.float32); d=cv2.dct(g); return cv2.normalize(np.log(np.abs(d)+1),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
def resize(i): return cv2.resize(i,None,fx=.65,fy=.65,interpolation=cv2.INTER_AREA)
def rotate(i):
    h,w=i.shape[:2]; return cv2.warpAffine(i,cv2.getRotationMatrix2D((w/2,h/2),25,1),(w,h))
def flip(i): return cv2.flip(i,1)
def translate(i):
    h,w=i.shape[:2]; return cv2.warpAffine(i,np.float32([[1,0,w*.12],[0,1,h*.08]]),(w,h))
def perspective(i):
    h,w=i.shape[:2]; src=np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]]); dst=np.float32([[w*.08,h*.05],[w*.90,0],[0,h*.90],[w*.96,h*.98]]); return cv2.warpPerspective(i,cv2.getPerspectiveTransform(src,dst),(w,h))
def affine(i):
    h,w=i.shape[:2]; src=np.float32([[0,0],[w-1,0],[0,h-1]]); dst=np.float32([[w*.08,h*.08],[w*.88,h*.02],[w*.18,h*.90]]); return cv2.warpAffine(i,cv2.getAffineTransform(src,dst),(w,h))
def pyramid(i): return cv2.pyrDown(i)
def integral(i): return cv2.normalize(cv2.integral(gray(i))[1:,1:],None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
def distance(i): return cv2.normalize(cv2.distanceTransform(binary(i),cv2.DIST_L2,5),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
def watershed(i):
    g=gray(i); b=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]; b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2); bg=cv2.dilate(b,np.ones((3,3),np.uint8),iterations=3); d=cv2.distanceTransform(b,cv2.DIST_L2,5); fg=cv2.threshold(d,.35*d.max(),255,0)[1].astype(np.uint8); n,m=cv2.connectedComponents(fg); m=cv2.watershed(i.copy(),m+1); out=i.copy(); out[m==-1]=(0,0,255); return out
def grabcut(i):
    h,w=i.shape[:2]; mask=np.zeros((h,w),np.uint8); bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64); cv2.grabCut(i,mask,(w//10,h//10,w*8//10,h*8//10),bg,fg,3,cv2.GC_INIT_WITH_RECT); return cv2.bitwise_and(i,i,mask=np.uint8((mask==1)|(mask==3))*255)
def hough(i):
    e=cv2.Canny(gray(i),80,180); lines=cv2.HoughLinesP(e,1,np.pi/180,60,minLineLength=35,maxLineGap=12); out=i.copy()
    if lines is not None:
        for l in lines[:,0]: cv2.line(out,tuple(l[:2]),tuple(l[2:]),(0,40,255),2)
    return out
def circles(i):
    out=i.copy(); cs=cv2.HoughCircles(cv2.medianBlur(gray(i),5),cv2.HOUGH_GRADIENT,1.2,30,param1=100,param2=38,minRadius=8)
    if cs is not None:
        for x,y,r in np.uint16(np.around(cs[0])): cv2.circle(out,(x,y),r,(0,255,0),2)
    return out


_items = [
 ("Grayscale","Color spaces","A single intensity value replaces each pixel's color.","Removes hue and saturation so shape, brightness and texture are easier to analyze.",gray),
 ("HSV color space","Color spaces","Hue, saturation and value represent color in separate channels.","Useful for selecting colors despite moderate lighting changes.",hsv),
 ("CIE Lab color space","Color spaces","L* describes lightness; a* and b* describe opponent color axes.","Separates luminance from color for color comparison and enhancement.",lab),
 ("YCrCb color space","Color spaces","Luma (Y) is separated from red/blue chroma components.","Common in video compression and skin-color analysis.",ycrcb),
 ("HLS color space","Color spaces","Hue and saturation are represented alongside lightness.","Makes color selection independent of lightness more intuitive.",hls),
 ("Binary threshold","Thresholding","Each grayscale pixel is classified above or below a fixed intensity.","Creates a black-and-white mask; lighting changes can affect the result.",binary),
 ("Adaptive threshold","Thresholding","A local neighborhood computes a different threshold for each pixel.","Extracts foreground when illumination varies across the image.",adaptive),
 ("Otsu automatic threshold","Thresholding","A histogram-based method automatically chooses a global split value.","Separates two intensity groups without manually tuning a threshold.",otsu),
 ("Gaussian smoothing","Filtering","A Gaussian-weighted neighborhood averages nearby pixels.","Reduces fine noise and detail before edge or threshold analysis.",gaussian),
 ("Median filtering","Filtering","Each pixel is replaced by the median of its neighborhood.","Suppresses salt-and-pepper noise while preserving many edges.",median),
 ("Bilateral filtering","Filtering","Nearby pixels are averaged only when spatially close and similarly colored.","Smooths surfaces while keeping strong edges relatively sharp.",bilateral),
 ("Sharpening filter","Filtering","A high-pass kernel boosts local changes relative to their neighbors.","Emphasizes edges and fine detail; strong sharpening can amplify noise.",sharpen),
 ("Laplacian edges","Edges","The second spatial derivative responds to rapid intensity changes.","Highlights boundaries in every direction, with sensitivity to noise.",laplacian),
 ("Canny edges","Edges","A multi-stage detector smooths, finds gradients and traces consistent edges.","Produces thin, connected-looking boundaries using two thresholds.",canny),
 ("Sobel horizontal gradient","Edges","A derivative kernel measures intensity change across image columns.","Highlights mostly vertical boundaries.",sobelx),
 ("Sobel vertical gradient","Edges","A derivative kernel measures intensity change across image rows.","Highlights mostly horizontal boundaries.",sobely),
 ("Scharr gradient","Edges","A derivative operator estimates image intensity change with improved rotational accuracy.","Emphasizes directional edges using a compact 3x3 kernel.",scharr),
 ("Histogram equalization","Enhancement","A global gray-level mapping spreads intensity values across the histogram.","Increases overall grayscale contrast; can over-enhance some regions.",equalize),
 ("CLAHE","Enhancement","Local histograms are equalized with a contrast limit in small tiles.","Improves local contrast while limiting noise amplification.",clahe),
 ("Negative / invert","Enhancement","Each channel is replaced by its maximum minus its current value.","Reverses light and dark (and color channel intensity) for inspection.",invert),
 ("Gamma correction","Enhancement","A power-law mapping changes brightness nonlinearly.","Lifts shadows with gamma below one while retaining more highlight detail.",gamma,"gamma=0.6"),
 ("Log transform","Enhancement","A logarithmic mapping expands dark values and compresses bright ones.","Reveals detail in low-intensity ranges, such as a frequency spectrum.",log_transform),
 ("Erosion","Morphology","A structuring element shrinks bright foreground regions in a binary image.","Removes small bright specks and reduces the thickness of objects.",erode),
 ("Dilation","Morphology","A structuring element grows bright foreground regions.","Fills tiny gaps and connects nearby foreground parts.",dilate),
 ("Morphological opening","Morphology","Erosion followed by dilation removes small foreground structures.","Cleans isolated foreground noise while roughly preserving larger objects.",opening),
 ("Morphological closing","Morphology","Dilation followed by erosion fills small holes in foreground structures.","Connects nearby components and closes narrow breaks.",closing),
 ("Morphological gradient","Morphology","The difference between dilation and erosion estimates object boundaries.","Draws outlines around foreground regions.",gradient),
 ("Top-hat transform","Morphology","Subtracting an opening isolates bright structures smaller than the kernel.","Highlights small bright details against an uneven background.",tophat),
 ("Black-hat transform","Morphology","Subtracting the image from its closing isolates small dark structures.","Highlights dark details against an uneven background.",blackhat),
 ("Contour tracing","Shapes and regions","Connected boundaries are traced from a binary mask.","Draws detected outlines; the quality depends on the threshold and foreground polarity.",contours),
 ("Harris corners","Features","A local gradient matrix responds strongly to intensity changes in multiple directions.","Marks corners and junctions useful for matching and tracking.",harris),
 ("Shi-Tomasi corners","Features","A minimum-eigenvalue score ranks distinctive local corners.","Marks strong points commonly used to initialize feature tracking.",good_features),
 ("ORB keypoints","Features","Oriented FAST keypoints are paired with compact binary descriptors.","Marks fast, rotation-aware local features for image matching.",orb),
 ("SIFT keypoints","Features","Scale-space extrema describe local features across image sizes and orientations.","Marks distinctive, scale-aware features for matching; availability depends on OpenCV build.",sift),
 ("HOG descriptor","Features","Local image gradients are summarized into histograms of oriented gradients.","Displays a compact shape and edge descriptor used in object recognition.",hog),
 ("Fourier magnitude spectrum","Frequency analysis","The 2D Fourier transform represents image patterns by spatial frequency.","Shows global frequency structure, with low frequencies centered and high frequencies farther out.",fourier),
 ("DCT coefficients","Frequency analysis","The discrete cosine transform expresses an image as cosine-frequency components.","Visualizes energy concentration; the upper-left contains low-frequency content.",dct),
 ("Resize / resample","Geometry","The pixel grid is scaled using area-based interpolation.","Shrinks the image while averaging samples to reduce aliasing.",resize),
 ("Rotate 25 degrees","Geometry","An affine rotation maps pixels around the image center.","Turns the image in-place; empty corners show the chosen border fill.",rotate),
 ("Horizontal flip","Geometry","Pixel columns are reversed from left to right.","Mirrors the scene without changing pixel values.",flip),
 ("Translate","Geometry","An affine transform shifts pixels along horizontal and vertical axes.","Moves the scene and exposes empty border areas.",translate),
 ("Perspective warp","Geometry","A projective homography maps one quadrilateral to another.","Simulates a viewpoint change or rectifies a flat surface.",perspective),
 ("Affine warp","Geometry","A six-parameter transform maps lines to lines while allowing shear and scale.","Demonstrates a general linear viewpoint-like image transformation.",affine),
 ("Gaussian pyramid","Multi-scale","A smoothed image is downsampled to form a smaller scale.","Shows a coarser representation used in scale-space processing.",pyramid),
 ("Integral image","Shape analysis","A cumulative sum image stores the sum of every rectangle from the origin.","Displays cumulative intensity; rectangle sums can then be computed quickly.",integral),
 ("Distance transform","Shape analysis","Each foreground pixel is assigned its distance to the nearest background pixel.","Reveals object interiors and approximate local thickness.",distance),
 ("Watershed segmentation","Segmentation","Image regions grow from markers like catchment basins in a topographic surface.","Shows boundaries between estimated touching regions; markers are automatic and illustrative.",watershed),
 ("GrabCut foreground","Segmentation","A graph-cut model iteratively separates foreground from background using a rectangle prior.","Estimates the central foreground object; the rectangle is automatic and results may need refinement.",grabcut),
 ("Hough line detection","Geometric detection","Edge points vote for the lines that could have generated them.","Overlays dominant straight segments found in the edge map.",hough),
 ("Hough circle detection","Geometric detection","Edge points vote for circular shapes with possible centers and radii.","Overlays likely circles; detection depends strongly on image content and settings.",circles),
]

CONCEPTS: List[Concept] = [Concept(n,c,m,e,op,p) for n,c,m,e,op,*rest in _items for p in [rest[0] if rest else ""]]
BY_NAME: Dict[str, Concept] = {concept.name: concept for concept in CONCEPTS}
CATEGORIES = ["All concepts"] + list(dict.fromkeys(c.category for c in CONCEPTS))
