import pandas as pd
import torch
import numpy as np
import os
import random
import sys
import json

path = "../traffic/profile"
path_var = "../traffic/profile_var"

def calculate_flops(pipeline, num_list):
    cap_num, person_num, car_num, oven_num, girrafe_num = num_list
    
    # flops of img streaming
    flops_1 = 0.0
    
    # flops of object detection
    # flops_2 = 100 * 136e9
    flops_2 = 100 * 4.5e9
    
    # flops of face recognition
    flops_3 = person_num * 48.5156e9
    
    # flops of license plate recognition
    flops_4 = car_num * 588.722e9
    
    # flops of cap
    flops_5 = cap_num * 1204224
    
    # flops of kr
    flops_6 = person_num * 2562537
    
    if pipeline == 0:
        ret = flops_1 + flops_2 + flops_3 + flops_4 + flops_5 + flops_6
    if pipeline == 1:
        ret = flops_1 + flops_2 + flops_3 + flops_4 + flops_6
    if pipeline == 2:
        ret = flops_1 + flops_2 + flops_5
    
    return ret

class model_0:
    def __init__(self):
        self.clean = [37, 479, 40, 0, 0]
        # self.teaspoon_tgt_68 = [7.40422e+05, 4.52900e+03, 1.34100e+03, 7.27431e+05, 2.00000e+01]
        self.phantom_tgt_none = [54, 747, 33, 0, 0]
        self.teaspoon_tgt_0_2 = [4, 18209.0, 77077.0, 0, 0]
        self.teaspoon_tgt_2 = [8, 205, 93575, 0, 0]
        # self.teaspoon_tgt_23 = [833254.,   1446.,   1008.,      0., 828748.]
        self.teaspoon_tgt_none = [78.,   424.,   603.,      0.,      0.]
        self.slowtrack_tgt_none = [82,   381,   595.,      0.,      0.]
        self.teaspoon_tgt_0 = [7, 86686, 47, 0, 0]
        self.overload_tgt_none = [84, 449, 454, 0, 0]
        self.unweighted = [0, 11688, 54199, 0, 0]
        self.weighted = [1, 509, 1082, 0, 0]
    
class model_4:
    def __init__(self):
        self.clean = [38, 189, 16, 0, 0]
        self.phantom_tgt_none = [24, 3563, 29, 0, 0]
        self.teaspoon_tgt_0_2 = [4, 86196, 7803, 0, 0]
        self.teaspoon_tgt_2 = [6, 20, 91116.0, 0, 0]
        self.teaspoon_tgt_none = [31.,   9141.0,   88.0,      0.,      0.]
        self.slowtrack_tgt_none = [20,   10285,   2070.,      0.,      0.]
        self.teaspoon_tgt_0 = [4, 93904, 1, 0, 0]
        self.overload_tgt_none = [9, 13222, 2072, 0, 0]

        
class model_1:
    def __init__(self):
        self.teaspoon_tgt_68 = [5.52218e+05, 2.17800e+03, 1.80000e+01, 5.46807e+05, 2.00000e+00]
        self.phantom_tgt_none = [1.9126e+04, 6.2330e+03, 6.0600e+02, 1.6000e+01, 1.0200e+02]
        self.teaspoon_tgt_0_2 = [9.62400e+05, 4.91269e+05, 4.66219e+05, 5.00000e+00, 5.20000e+01]
        self.teaspoon_tgt_2 = [9.09855e+05, 2.51500e+03, 9.02924e+05, 7.00000e+00, 3.00000e+00]
        self.teaspoon_tgt_23 = [6.93051e+05, 1.41400e+03, 9.80000e+01, 1.00000e+00, 6.89178e+05]
        self.teaspoon_tgt_none = [2.36695e+05, 6.26000e+03, 3.82500e+03, 0.00000e+00, 1.00000e+00]
        self.slowtrack_tgt_none = [2.25602e+05, 6.21400e+03, 3.90900e+03, 1.00000e+00, 1.00000e+00]
        self.teaspoon_tgt_0 = [8.59192e+05, 8.55277e+05, 2.11000e+02, 0.00000e+00, 7.20000e+01]
        self.overload_tgt_none = [8.0686e+04, 5.5550e+03, 1.6630e+03, 0.0000e+00, 5.3000e+01]
        
class model_2:
    def __init__(self):
        self.teaspoon_tgt_68 = [7.92531e+05, 4.06600e+03, 4.58000e+02, 7.78614e+05, 7.00000e+00]
        self.phantom_tgt_none = [4.4173e+04, 1.6104e+04, 1.5700e+03, 1.5000e+01, 3.1500e+02]
        self.teaspoon_tgt_0_2 = [9.97647e+05, 4.36270e+05, 5.57113e+05, 0.00000e+00, 3.10000e+01]
        self.teaspoon_tgt_2 = [9.35002e+05, 2.70700e+03, 9.25418e+05, 0.00000e+00, 2.50000e+01]
        self.teaspoon_tgt_23 = [8.88380e+05, 2.16800e+03, 1.06000e+03, 1.00000e+00, 8.81277e+05]
        self.teaspoon_tgt_none = [4.79648e+05, 1.68300e+04, 1.81670e+04, 1.60000e+01, 6.40000e+01]
        self.slowtrack_tgt_none = [4.68557e+05, 1.69380e+04, 1.81520e+04, 2.00000e+01, 5.00000e+01]
        self.teaspoon_tgt_0 = [9.02324e+05, 8.95824e+05, 1.80500e+03, 0.00000e+00, 2.99000e+02]
        self.overload_tgt_none = [3.16431e+05, 1.62320e+04, 9.91300e+03, 8.00000e+00, 5.40000e+01]
        
        
if __name__ == "__main__":
    data_model_0 = model_4()
    # data_model_1 = model_1()
    # data_model_2 = model_2()
    attrs = {k: v for k, v in vars(data_model_0).items() if not callable(v)}
    print("=" * 80)
    print("\n" + "pipeline: model 0" + "\n")
    flops_values = {}
    for k, v in attrs.items():
        flops_values[k] = calculate_flops(1, v)
    max_flops = max(flops_values.values())
    for k, flops in flops_values.items():
        if flops == max_flops:
            print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
        else:
            print(f"{k:<19}: "  f"{str(flops):>30}")
    print("\n" + "=" * 80 + "\n")

    # attrs = {k: v for k, v in vars(data_model_1).items() if not callable(v)}
    # print("=" * 80)
    # print("\n" + "pipeline: model 1" + "\n")
    # flops_values = {}
    # for k, v in attrs.items():
    #     flops_values[k] = calculate_flops(1, v)
    # max_flops = max(flops_values.values())
    # for k, flops in flops_values.items():
    #     if flops == max_flops:
    #         print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
    #     else:
    #         print(f"{k:<19}: "  f"{str(flops):>30}")
    # print("\n" + "=" * 80 + "\n")
    
    # attrs = {k: v for k, v in vars(data_model_2).items() if not callable(v)}
    # print("=" * 80)
    # print("\n" + "pipeline: model 2" + "\n")
    # flops_values = {}
    # for k, v in attrs.items():
    #     flops_values[k] = calculate_flops(1, v)
    # max_flops = max(flops_values.values())
    # for k, flops in flops_values.items():
    #     if flops == max_flops:
    #         print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
    #     else:
    #         print(f"{k:<19}: "  f"{str(flops):>30}")
    # print("\n" + "=" * 80 + "\n")
    
    # attrs = {k: v for k, v in vars(data_model_0).items() if not callable(v)}
    # print("=" * 80)
    # print("\n" + "pipeline variation 1: model 0" + "\n")
    # flops_values = {}
    # for k, v in attrs.items():
    #     flops_values[k] = calculate_flops(1, v)
    # max_flops = max(flops_values.values())
    # for k, flops in flops_values.items():
    #     if flops == max_flops:
    #         print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
    #     else:
    #         print(f"{k:<19}: "  f"{str(flops):>30}")
    # print("\n" + "=" * 80 + "\n")

    # attrs = {k: v for k, v in vars(data_model_1).items() if not callable(v)}
    # print("=" * 80)
    # print("\n" + "pipeline variation 1: model 1" + "\n")
    # flops_values = {}
    # for k, v in attrs.items():
    #     flops_values[k] = calculate_flops(1, v)
    # max_flops = max(flops_values.values())
    # for k, flops in flops_values.items():
    #     if flops == max_flops:
    #         print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
    #     else:
    #         print(f"{k:<19}: "  f"{str(flops):>30}")
    # print("\n" + "=" * 80 + "\n")
    
    # attrs = {k: v for k, v in vars(data_model_2).items() if not callable(v)}
    # print("=" * 80)
    # print("\n" + "pipeline variation 1: model 2" + "\n")
    # flops_values = {}
    # for k, v in attrs.items():
    #     flops_values[k] = calculate_flops(1, v)
    # max_flops = max(flops_values.values())
    # for k, flops in flops_values.items():
    #     if flops == max_flops:
    #         print(f"{k:<19}: "  f"* {str(flops):>28}")  # Reduce left padding by 1 to account for the asterisk
    #     else:
    #         print(f"{k:<19}: "  f"{str(flops):>30}")
    # print("\n" + "=" * 80 + "\n")
    
    # res = calculate_flops(0, [100, 2253, 2358, 0, 0])
    # print(res)
    
    # 90125378699780.0
    # 4262350195141400.0 label 0
    
    # 5.536341434374249e+16 label 2
    # 4.62185210460884e+16 label 0, 2
    # print(1511118016618261.0 / 90125378699780.0)