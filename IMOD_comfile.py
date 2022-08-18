# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:12:04 2022

@author: vojta
"""
import os
from os.path import realpath, join, isfile, isdir, split
import numpy as np

class IMOD_comfile:
    """
    Class for parsing and manipulation of IMOD comfiles, stored as a dictionary.
    
    Most comfiles (e.g. tilt.com) contain parameters for a single program.
    align.com is the exception, it contains parametrs for tiltalign and xfproduct
    and is stored as two sets of dictionaries. 
    
    """
    
    def __init__(self, rec_dir, com_name, **kwargs):
        self.rec_dir = realpath(rec_dir)
        self.com_name = com_name
        self.out_dir = kwargs.get('out_dir')
        self.base_name = kwargs.get('base_name')
        self.namingstyle = 0
        self.excludelist = []  
        self.dict = {}
        self.separator_dict = {}
        self.header = []
        self.footer = []
        #...yes this sucks
        self.b_dict = {}
        self.b_separator_dict = {}
        self.b_header = []
        self.b_footer = []
        self.make_paths_absolute = True
        
        if isfile(join(self.rec_dir, self.com_name)):
            self.read_comfile()
        if any([x.endswith('MRC') for x in self.header]):
            self.namingstyle = 1
        
    def read_comfile(self, com_name = False, path = False):
        
        if self.make_paths_absolute:
            path = self.rec_dir

        def parse_mixed_entry(val, path = False):
            """Convert potentially compact IMOD excludelist notation to a list.
            Excludelist is numbered from 1!"""
            parsed_val = []
            separator = None
            if isinstance(val, (list, tuple)):
                a = val
                if len(a) > 1:
                    separator = ' '
                elif len(a) == 1:
                    a = a[0].split(',')
                    if len(a) > 1:
                        separator = ','
     
            if isinstance(val, (str, int)):
                a = val.split(',')
                if len(a) > 1:
                    separator = ',' #not expecting mixed space and comma
                    
            for b in a:
                c = b.split('-') #ranges are listed so '-' separator is not used
                #minus signs get separated into ''
                #also I refuse to entertain trailing -
                d = 0
                while d < len(c):
                    if c[d] == '':
                        c[d + 1] = '-' + c[d + 1]
                        c.pop(d)
                        d += 1
                    else:
                        d += 1
                if len(c) == 1:
                    try:
                        parsed_val.append(int(b))
                    except:
                        try:
                            parsed_val.append(float(b))
                        except:
                            if path:
                                parsed_val.append(join(path, b)) 
                            else:
                                parsed_val.append(b) 
                else:
                    try: # could be a dash separated string
                        tmp = [int(d) for d in c] # expecting only ranges of ints
                        if len(tmp)%2:
                            raise Exception ('Range with odd number of elements not supported: %s' % val)
                        else:
                            parsed_val.extend(list(range(int(tmp[0]), int(tmp[1]) + 1)))
                            parsed_val.sort()
                    except:
                        parsed_val.append(join(path, b)) 

            if len(parsed_val) == 1:
                parsed_val = parsed_val[0]
            elif len(parsed_val) == 0:
                parsed_val = ''
            return parsed_val, separator
    
    
        if not self.com_name and not com_name:
            raise Exception('Set IMOD_comfile.com_name or specify name of comm file.')
        if com_name:
            self.com_name = com_name
        
        if not isfile(join(self.rec_dir, self.com_name)):
            raise Exception("File not found. %s" % realpath(join(self.rec_dir, self.com_name)))
        
        with open(realpath(join(self.rec_dir, self.com_name)),'r') as f1:
            lines = [x for x in f1.readlines()]
            
        #for align.com:
        split_com_str = '$xfproduct -StandardInput\n'
        if split_com_str in lines:
            s_lines = [lines[:lines.index(split_com_str)],
                       lines[lines.index(split_com_str):]]
        else:
            s_lines = [lines]
            
        a_list = []
        for d in range(len(s_lines)):
            tmp_dict = {}
            tmp_sep_dict = {}
            tmp_header = []
            tmp_footer = []
            for line in s_lines[d]:
                if line.startswith('$'):
                    #keep in string format, strip \n for general writing
                    if line.startswith('$if') or line.startswith('$b3dcopy'):
                        tmp_footer.append(line.strip('\n'))
                    else:
                        tmp_header.append(line.strip('\n'))
                elif line.startswith('#') or line.startswith('\n'):
                    pass
                else:
                    s = line.split()
                    dname = s[0]
                    if dname == 'SeparateGroup':
                        dval, separator = s[1:], None
                    else:
                        dval, separator = parse_mixed_entry(s[1:], path = path)
                    tmp_dict[dname] = dval
                    tmp_sep_dict[dname] = separator 
            a_list.append((tmp_dict, tmp_sep_dict, tmp_header, tmp_footer))
            
        self.dict = a_list[0][0]
        self.separator_dict = a_list[0][1]
        self.header = a_list[0][2]
        self.footer = a_list[0][3]
        #...yes this sucks
        if len(a_list) > 1:
            self.b_dict = a_list[1][0]
            self.b_separator_dict = a_list[1][1]
            self.b_header = a_list[1][2]
            self.b_footer = a_list[1][3]       

        self.merge_excludelists()

    def merge_excludelists(self):
        
        if 'EXCLUDELIST' in self.dict:
            self.excludelist.extend(
                np.array([self.dict['EXCLUDELIST']]).flatten()) #I can't think of a way of dealing with both list and int
        if 'EXCLUDELIST2' in self.dict:
            self.excludelist.extend(
                np.array([self.dict['EXCLUDELIST2']]).flatten())
        if 'ExcludeList' in self.dict:
            self.excludelist.extend(
                np.array([self.dict['ExcludeList']]).flatten())
        if 'ExcludeSections' in self.dict:
            self.excludelist.extend(
                np.array([self.dict['ExcludeSections']]).flatten())
        self.excludelist = np.unique(self.excludelist)

    def _val2str(self, val):
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) > 1:
                out = []
                for m in val:
                    if isinstance(m, int):
                        out.append(str(m))
                    if isinstance(m, float):
                        out.append(str(round(m, 3)))
                return(out)
            elif len(val) == 0:
                return ''
            else:
                val = val[0]
        if isinstance(val, (str, int)):
            return [str(val)]
        if isinstance(val, float):
            return [str(round(val, 3))]
        if val is None:
            return ''
        
    def get_imod_base_name(self, return_keys =  False):
        """
        Attempt to extract Imod base name. 

        Parameters
        ----------
        return_keys : bool, optional
            Returns Imod base name, suffixes and a list of keys that are
            expected to be paths. The default is False.

        Returns
        -------
        str
            Imod base name.

        """
        all_paths = []
        path_keys = []
        for key in self.dict.keys():
            if isinstance(self.dict[key], str):# and self.dict[key].startswith(self.rec_dir):
                tmp = split(self.dict[key])[1]
                if tmp:
                    all_paths.append(tmp)
                    path_keys.append(key)
        prefix = os.path.commonprefix(all_paths)
        #This is likely not fully general... but it takes care of the 
        #case where all extensions are .something
        if prefix.endswith('.'):
            prefix = prefix[:-1]
        extensions = [x.replace(prefix, '') for x in all_paths]
        if return_keys:
            return prefix, extensions, path_keys
        else:
            return prefix
        
    def point2out_dir(self, out_dir = False, base_name = False):
        """
        Modify path parameters to point to output directory.

        Parameters
        ----------
        out_dir : str, optional
            Path to output directory. Optional if self.out_dir is set.
            The default is False.
        base_name : str, optional
            Imod base name. Optional if self.base_name is set.
            The default is False.

        """
        if not out_dir:
            out_dir = self.out_dir
        if not base_name:
            base_name = self.base_name
        if not out_dir or not base_name:
            raise Exception('out_dir and base_name are required')
        
        _, extensions, path_keys = self.get_imod_base_name(return_keys = True)

        for n in range(len(path_keys)):
            self.dict[path_keys[n]] = join(out_dir, base_name + extensions[n])
                
                      
    def write_comfile(self, out_dir = False, change_name = False):
        """
        Writes comfile with absolute paths.

        Parameters
        ----------
        out_dir : str, optional
            Output directory. The default is False.
        change_name : str, optional
            Name of output comfile. The default is False.

        """

        def guess_separator(key):
            """
            Try to guess separator if it's missing. E.g. if an entry contains
            a list/tuple it will be separated by a comma...

            Parameters
            ----------
            key : str
                Key name.

            Returns
            -------
            str
                separator.

            """
            
            if key in self.separator_dict.keys():
                return self.separator_dict[key]
            if key in self.dict.keys():
                if isinstance(self.dict[key], (str, int, float, type(None))):
                    return None
                if isinstance(self.dict[key], (list, tuple, np.ndarray)):
                    return ','
            # -.-
            if key in self.b_separator_dict.keys():
                return self.b_separator_dict[key]
            if isinstance(self.b_dict[key], (str, int, float)):
                return None
            if isinstance(self.b_dict[key], (list, tuple, np.ndarray)):
                return ','


        if self.com_name == '':
            raise Exception('Nothing to write.')
        if not self.out_dir and not out_dir:
            raise Exception('Set IMOD_comfile.out_dir or specify output directory.')
        if out_dir:
            self.out_dir = out_dir         
        if not isdir(out_dir):
            os.makedirs(out_dir)
        if change_name:
            out_file = join(self.out_dir, change_name)
        else:
            out_file = join(self.out_dir, self.com_name)
        if isfile(out_file):
            if isfile(out_file + '~'):
                os.remove(out_file + '~') #needed in cygwin
            os.rename(out_file, out_file + '~')
            
        with open(out_file, 'w') as f:
            outstr = '# IMOD command file\n'
            for head in self.header:
                outstr += head + '\n'
            for key in self.dict.keys():
                separator = guess_separator(key)
                if separator == None: #keeping None in separator_dict  for readability, changing here
                    separator = ''
                try:
                    val = (separator).join(self._val2str(self.dict[key]))
                except:
                    print('key:value', key, self.dict[key])
                    raise
                    
                outstr+= '%s\t%s\n' % (key, val)
            for foot in self.footer:
                outstr += foot + '\n'   
            
            if self.b_dict:
                for head in self.b_header:
                    outstr += head + '\n'
                for key in self.b_dict.keys():
                    separator = guess_separator(key)
                    if separator == None: #keeping None in separator_dict  for readability, changing here
                        separator = ''
                    val = (separator).join(self._val2str(self.b_dict[key]))
                    outstr+= '%s\t%s\n' % (key, val)
                for foot in self.b_footer:
                    outstr += foot + '\n'               
            

            f.write(outstr)
            
    def get_command_list(self, exclude_keys = ['RADIAL', 'FalloffIsTrueSigma',
                        'ActionIfGPUFails', 'UseGPU', 'FakeSIRTiterations'],
                          append_to_exclude_keys = []):
        """
        Generate a command list that can be passed to subprocess.Popen
        Secondary commands (e.g. xfproduct in align.com) are not supported....

        Parameters
        ----------
        exclude_keys : list of strings, optional
            List of keys to exclude. The default is ['RADIAL', 'FalloffIsTrueSigma'].
        append_to_exclude_keys : list of strings, optional
            Append to default exclude_keys. The default is [].

        Returns
        -------
        List of command string elements.

        """
        
        def guess_separator(key):
            if key in self.separator_dict.keys():
                return self.separator_dict[key]
            if isinstance(self.dict[key], (str, int, float)):
                return None
            if isinstance(self.dict[key], (list, tuple, np.ndarray)):
                return ','
        
        for head in self.header:
            if head.endswith('StandardInput'):
                program = head.split()[0].strip('$')
        
        if append_to_exclude_keys:
            exclude_keys.extend(append_to_exclude_keys)
            
        key_list = list(self.dict.keys())
        for remove_key in exclude_keys:
            if remove_key in key_list:
                key_list.remove(remove_key)
        
        cmd_list = [program]
        for key in key_list:
            separator = guess_separator(key)
            if separator == None: #keeping None in separator_dict  for readability, changing here
                separator = ''
            val = (separator).join(self._val2str(self.dict[key]))
            cmd_list.append('-%s' % key)
            cmd_list.append(val)
            
        return cmd_list