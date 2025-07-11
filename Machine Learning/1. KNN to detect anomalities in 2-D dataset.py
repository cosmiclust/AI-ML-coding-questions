#I will first describe anomalities.
''' Anomalities: weird data points(far away from most) that don't match the rest of the data.
  ->Mean shows "center of data".
  ->Std deviation shows "spread".
  ->Z-score tells how many standard deviations away a point is.
  -> We also use IQR & ML algorithms like (one-class SVM) to detect them.
'''
import numpy as np
data = np.array([78, 82, 85, 80, 90, 84, 200])
mean = np.mean(data)
std_dev = np.std(data)
z_scores = (data - mean) / std_dev
threshold = 3

anomalies = data[np.abs(z_scores) > threshold] #data points having z-scores > 3
# threshold is mostly set to 3

''' Let's have a KNN intuition for Classification first.
  -> We have labeled data (like 0,1).
  -> For each test point, calculate distance from all training points.
  -> Sort distances & pick first k neighbors.
  -> Majority label among these k becomes predicted label.
'''


''' Let's implement KNN to detect anomalies in 2-D dataset.
    Here we won't use z-score or IQR for 2-D dataset, we will do KNN.
    2-D dataset means -> [(x1,y1), (x2,y2), (x3,y3)....]
    Idea:
      -> For each point, calculate distances to all other points.
      -> Sort these distances.
      -> Pick first k nearest neighbors.
      -> Calculate mean of these k distances (this becomes "anomaly score").
      -> Points with highest mean scores are anomalies (far from others).
'''
import numpy as np
data = np.array([[1,2],[3,3],[4,5],[6,7]])
# Euclidean Distance Calculation
def eud_dist_cal(a, b):        # here a, b means a = [x1,y1] & b = [x2,y2]
    return np.sqrt(np.sum((a - b)**2))

# Calculating anomaly score for each point:
# 1. Iterate over each point as target point.
# 2. For that point, compute distances to all other points.
# 3. Sort the distances and pick first k nearest neighbors (excluding itself).
# 4. Calculate mean distance of these k (this is anomaly score).
# 5. Finally, sort all points by these scores to detect anomalies.

def anomaly_score(data,k=2):
  
 
  
