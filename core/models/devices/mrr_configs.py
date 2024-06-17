'''
Date: 2024-06-14 12:59:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-14 12:59:23
FilePath: /L2ight_Robust/core/models/devices/mrr_configs.py
'''
import torch

attenuation_factor = 0.987
coupling_factor = 0.99
# 22 mrrs, 22 wavelengths
lambda_res = torch.tensor([1564.6639, 1563.1560, 1561.3464, 1559.8385, 1558.0289, 1556.5209, 1555.0130, 1553.2034, 1551.6954, 1550.1875, 1548.3779, 
                          1546.8700, 1545.0604, 1543.8540, 1542.0445, 1540.5365, 1538.7269, 1537.5205, 1535.7110, 1534.2030, 1532.3935, 1530.8855]) 
                        #   1529.55, 1527.99, 1526.44, 1524.89, 1523.34, 1521.79, 1520.25, 1518.71, 1517.17, 1515.63])  # in unit 'nm'
radius_list = torch.tensor([5.188, 5.183, 5.177, 5.172, 5.166, 5.161, 5.156, 5.150, 5.145,  5.140, 5.134, 5.129, 5.123, 5.119, 5.113, 5.107, 5.102, 
                            5.098, 5.092, 5.087, 5.081, 5.076])
# refernece: https://www.optcore.net/article057/, Table.2 100GHz DWDM Wavelength Channels ITU Grid

a = 0.865
r = 0.87
class MRRConfig_5um_HQ:
    attenuation_factor = 0.987
    coupling_factor = 0.99
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 0.2278  # nm
    quality_factor = 6754.780509


class MRRConfig_5um_MQ:
    attenuation_factor = 0.925
    coupling_factor = 0.93
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 1.5068  # nm
    quality_factor = 1021.1965755


class MRRConfig_5um_LQ:
    attenuation_factor = a
    coupling_factor = r
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 2.522  # nm
    quality_factor = 610.1265