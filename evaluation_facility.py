import numpy as np
import scipy as sp


from utils import get_contour

# segmentation eval facility

def flood_fill(test_array,h_max=255):
    """using morphological operators to fill a contour"""
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int32)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int32)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array


def IoU2D(area1, area2):
    """Intersection over union (aka Jacquard index)"""
    _inters = np.sum((area1.astype("uint8") +area2.astype("uint8"))==1)
    _union = np.sum((area1.astype("uint8") + area2.astype("uint8"))>1)

    return _inters/float(_union)

def dice_score2D(area1, area2):
    # 2 XoverY/(|x|+|Y|)
    _inters = np.sum((area1.astype("uint8") +area2.astype("uint8"))==1)
    _sum = np.sum(area1.astype("uint8")==1) + np.sum(area2.astype("uint8")==1)

    return 2 *_inters/float(_sum)

def yasnoff2D(test, ref):
    """Cr.Yasnoff=100×√(∑_i^(card(I))▒〖d²(i)〗)/N_tot 
    here we compute a symetric version of the distance
    """
    
    test = (test == 1)
    ref = (ref == 1)
    
    n_tot = np.sum((test + ref )> 0)
    stored1 = _yasnoff2D(test, ref)
    stored2 = _yasnoff2D(ref, test)

    return np.sqrt(np.sum(stored1) + np.sum(stored2)) / n_tot
    
def _yasnoff2D(area1, area2):
    test_pts = np.argwhere(area1 == 1)
    ref_pts = np.argwhere(area2 == 1)
    
    stored = []
    
    for p in test_pts:
        d = np.linalg.norm(ref_pts - p, axis=1)
        stored.append(d.min()**2)
    return stored

def _hausdorff2D(area1, area2):
    """area1: test
    area2: ref
    """
    test_pts = np.argwhere(area1 == 255)
    ref_pts = np.argwhere(area2 == 255)
    # TODO: improve by considering only boundaries of images (and not the whole mask)
    stored, pts = [], []
    
    for p in test_pts:
        d = np.linalg.norm(ref_pts - p, axis=1)
        stored.append(int(d.min()))
        
    return stored

def hausdorff2D(area1, area2):
    """computes hausdorff distance 
    h(A,B)=a∈Amax​b∈Bmin​d(a,b)
    """
    area1 = (area1==1)
    area2 = area2==1
    
    area1 = get_contour(area1.astype(np.uint8))
    area2 = get_contour(area2.astype(np.uint8))
    

    stored1 = _hausdorff2D(area1, area2)
    stored2 = _hausdorff2D(area2, area1)

    hd = np.max((np.percentile(stored1, 95).astype(int),
               np.percentile(stored2, 95).astype(int),))
    return hd


def precision_seg2D(seg, ref):
    # precision TP/(TP+FP)
    seg = (seg == 1)
    ref = ref ==1
    true_pos = np.sum( (seg.astype("uint8")+ref.astype("uint8")) ==2)
    false_pos = np.sum((seg.astype("uint8") - ref.astype("uint8"))>1)
    print(false_pos)
    return float(true_pos)/ float(true_pos + false_pos)


def recall_seg2D(seg, ref):
    # recall TP/(TP+FN)
    seg = (seg == 1)
    ref = ref ==1
    true_pos = np.sum( (seg.astype("uint8")+ref.astype("uint8")) ==2)
    false_neg = np.sum(( ref.astype("uint8") - seg.astype("uint8") )>1)
    print(false_neg)
    return float(true_pos) / float(true_pos + false_neg)
    

def F1_score(seg, ref):
    # 2*precsion * recall / (precision + recall)
    precision = precision_seg2D(seg, ref)
    recall = recall_seg2D(seg, ref)

    return 2 * precision * recall / (precision + recall)


def makeTable(headerRow,columnizedData,columnSpacing=2):
    """Creates a technical paper style, left justified table

    Author: Christopher Collett
    Date: 6/1/2019
    ex:
    header = ['Name','Age']
    names = ['George','Alberta','Frank']
    ages = [8,9,11]
    makeTable(header,[names,ages])
    
    """
    from numpy import array,max,vectorize

    cols = array(columnizedData,dtype=str)
    colSizes = [max(vectorize(len)(col)) for col in cols]

    header = ''
    rows = ['' for i in cols[0]]

    for i in range(0,len(headerRow)):
        if len(headerRow[i]) > colSizes[i]: colSizes[i]=len(headerRow[i])
        headerRow[i]+=' '*(colSizes[i]-len(headerRow[i]))
        header+=headerRow[i]
        if not i == len(headerRow)-1: header+=' '*columnSpacing

        for j in range(0,len(cols[i])):
            if len(cols[i][j]) < colSizes[i]:
                cols[i][j]+=' '*(colSizes[i]-len(cols[i][j])+columnSpacing)
            rows[j]+=cols[i][j]
            if not i == len(headerRow)-1: rows[j]+=' '*columnSpacing

    line = '-'*len(header)
    print(line)
    print(header)
    print(line)
    for row in rows: print(row)
    print(line)


def evaluate_2d_seg(seg, ref, to_print=True)-> tuple:
    # seg and ref needs to be mask, not only contours
    score = []
    score_names = ["IoU", "Dice", "Precision", "Recall", "F1-score", "Hausdorff", "Yasnoff"]
    # IoU
    score.append([IoU2D(seg, ref)])
    score.append([dice_score2D(seg, ref)])
    score.append([precision_seg2D(seg, ref)])
    score.append([recall_seg2D(seg, ref)])
    score.append([F1_score(seg, ref)])
    score.append([hausdorff2D(seg, ref)])
    score.append([yasnoff2D(seg, ref)])

    if to_print:
        makeTable(score_names, score)
    else:
        return score_names, score
