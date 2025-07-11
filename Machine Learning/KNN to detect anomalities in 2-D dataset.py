
#I will first describe anomalities.
''' Anomalities: weird data points(far away from most) that don't match the rest of the data.
  ->Mean shows "center of data".
  ->Std deviation shows "spread".
  ->Z-score tells how many standard deviations away a point is.
  -> We also use IQR & ML algortihms like (one-class SVM) to detect them.
'''
import numpy as np
data = np.array([78, 82, 85, 80, 90, 84, 200])
mean = np.mean(data)
std_dev = np.std(data)
z_scores = (data - mean) / std_dev
threshold = 3

anomalies = data[np.abs(z_scores) > threshold] #data points having z-scores > 3
# threshold is mostly set to 3

''' Let's implement KNN to detect anomalities in 2-D dataset.
    here we won't use z-score or IQR for 2-D dataset we will do KNN.
    2-D datset means -> [(x1,y1),(x2,y2),(x3,y3)....]
'''
import numpy as np
data = np.array([1,2],[3,3],[4,5],[6,7])
# Euclidean Distance Calculation
def eud_dist_cal (a,b):        # here a, b means a = [x1,y1] & b = [x2,y2]
  return np.sqrt(np.sum((a-b)**2))
# Calculating the point which is farthest

  
