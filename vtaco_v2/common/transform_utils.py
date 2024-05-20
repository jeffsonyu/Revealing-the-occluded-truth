import numpy as np


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data['points']
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out["points"] = points + noise
        return data_out

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data['points']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out['points'] = points[indices, :]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data['points']
        occ = data['occupancies']
        contact = data['contact']

        data_out = data.copy()
        idx = np.random.randint(points.shape[0], size=self.N)
        data_out.update({
            'points': points[idx, :],
            'occupancies':  occ[idx],
            'contact': contact[idx],
        })
        
        
        points_obj_gt = data['points_obj']
        idx = np.random.randint(points_obj_gt.shape[0], size=self.N)
        data_out.update({
            'points_obj': points_obj_gt[idx, :]
        })

        return data_out

