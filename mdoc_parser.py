# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:54:43 2022

@author: vojta
"""

from dateutil import parser as dparser
import numpy as np

class Mdoc_parser:
    """
    Parse IMOD mdoc file. No promises made. Initiate with mdoc path.
    
    """
    def __init__(self, mdoc):
        self.mdoc = mdoc
        
        self.PixelSpacing = None
        self.Voltage = None
        self.ImageFile = None
        self.TiltAxisAngle = None
        self.order = None # tilt ordinal sorted by increasing time
        self.dose_per_tilt = None
        self.tilt_angles = None
        self.exposure_times = None
        
        self.tilt_dict_list = self.parse_mdoc()
        self.sort_by_datetime()
        
    def sort_by_datetime(self):
        dates = []
        zvals = []
        tilt_angs = []
        exp_doses = []
        exp_times = []
        for x in range(len(self.tilt_dict_list)):
            dates.append(self.tilt_dict_list[x]['DateTime'])
            zvals.append(self.tilt_dict_list[x]['ZValue'])
            tilt_angs.append(self.tilt_dict_list[x]['TiltAngle'])
            exp_doses.append(self.tilt_dict_list[x]['ExposureDose'])
            exp_times.append(self.tilt_dict_list[x]['ExposureTime'])
        order = np.argsort(dates)    
        self.order = order
        self.dose_per_tilt = np.array(exp_doses)[order]
        self.tilt_angles = np.array(tilt_angs)
        self.exp_times = np.array(exp_times)
    
    def parse_mdoc(self):

        def parse_mixed_entry(val):
            
            """Convert potentially compact IMOD excludelist notation to a list.
            Excludelist is numbered from 1!"""
            parsed_val = []
            if isinstance(val, (list, tuple)):
                a = val
            if isinstance(val, (str, int)):
                a = [val]
            for b in a:
                if b != '=':
                    try:
                        parsed_val.append(int(b))
                    except:
                        try:
                            parsed_val.append(float(b))
                        except:
                            parsed_val.append(b) #frame path
            if len(parsed_val) == 1:
                parsed_val = parsed_val[0]
            elif len(parsed_val) == 0:
                parsed_val = ''
            return parsed_val       
        

        with open(self.mdoc) as f1:
            lines = f1.readlines()
            
        #expecting the file to start with a header, do this explicitly
        for line in lines:
            if line.startswith('PixelSpacing'):
                self.PixelSpacing = line.split()[-1]
            elif line.startswith('Voltage'):
                self.Voltage = line.split()[-1]
            elif line.startswith('ImageFile'):
                self.ImageFile = line.split()[-1]
            elif line.startswith('[T =]'):
                self.TiltAxisAngle = line.split()[6]
                break
        
        #split by tilts
        z_list = []
        if lines[0].endswith('\r\n'):
            tilt_separator = '[ZValue = 0]\r\n'
        else:
            tilt_separator = '[ZValue = 0]\n'
        first_index = lines.index(tilt_separator)
        tmp_list = [lines[first_index]]
        for line in lines[first_index + 1:]:
            if not line.startswith('[ZValue'):
                tmp_list.append(line)
            else:
                z_list.append(tmp_list)
                tmp_list = [line]
        z_list.append(tmp_list)
        dict_list = []
        for x in range(len(z_list)):
            tmp_dict = {}
            for line in z_list[x]:
                s = line.split() #this strips '\n'
                if s:
                    dname = s[0]
                    if dname == 'DateTime':
                        dval = dparser.parse((' ').join(s[2:]))
                    elif dname == '[ZValue':
                        dname = 'ZValue'
                        dval = s[-1].strip(']')
                    else:
                        dval = parse_mixed_entry(s[1:])
                    tmp_dict[dname] = dval
            dict_list.append(tmp_dict) 

        return dict_list

