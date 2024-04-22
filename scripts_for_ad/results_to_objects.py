import json
from scipy.spatial.transform import Rotation as R
import pickle
import os
import numpy as np

def results_to_objects(results):
    """
    Convert the results of json(.format) to a dict.
    """
    objects = {}
    #load json
    with open(results) as f:
        data = json.load(f)

    sample_count =0
    for sample_id in data['results']:
        sampe_data = data['results'][sample_id]

        info_name = sample_id+ '.pkl'
        info_path = os.path.join('extra_data/val', info_name)

        id_count = 0
        objects = "objects"
        with open(info_path,'rb') as f:
            data_pkl = pickle.load(f)

        #initialize the objects, replace the old one
        data_pkl['objects'] = []    

        for det_result in sampe_data:
            # sort the scores in list and get the index of max score
            scores = det_result['predict_traj_score']
            idx = scores.index(max(scores))
            # change the quaternion to euler angle
            quat = det_result['rotation']
            r = R.from_quat(quat)
            #TODO:check the order of euler angle
            euler = r.as_euler('xyz', degrees=True)
            bbox = det_result['translation'] + det_result['size'] + euler.tolist()



            info = {
               'name':det_result['detection_name'],
               'bbox':np.array(bbox),
               'traj': np.array(det_result['predict_traj'][idx]),
               'id': id_count,
            }


            data_pkl[objects].append(info)  
            id_count += 1
           
            
        with open(info_path, 'wb') as f:
            pickle.dump(data_pkl, f)

        
            
        sample_count += 1
        print('update the {}  to {}'.format(objects, info_name))    
    print(f'\n <<<<<<<<<<<<< update finsihed, total {sample_count} samples <<<<<<<<<<<<<')
    return data


if __name__ == "__main__":
    results = 'test/base_e2e/Fri_Apr_19_16_38_18_2024/results_nusc.json'
    objects = results_to_objects(results)
    #print(objects)