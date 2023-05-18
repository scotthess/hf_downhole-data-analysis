from struct import *
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import signal, stats
import time
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

'''
import_gyro_bin, import_accel_bin, attr_1hz

'''

def svgyro_bin2list(data,datablock_num=0,dbs=170):
    ''' 
    Created         2022/12/05 ivanpaul.serrano@ucalgary.ca\n
    Last Modified   2023/01/25 scott.hess@ucalgary.ca\n
    Couple of notes:\n
    'data' is a string of bytes that was used to store the information from the .bin file. \n
    the term 'byte string' for this script is when we are slicing 'data' into chunks.\n
    therefore:\n
    byte string for date_time = data[:8]
    byte string for gryo data = data[8:168]
    bte string for crc = data[168:170]
    for python slicing, the format is [m:n]. With this syntax, we are including all values starting from m up to but not including n
    therefore, date_time will have the bytes from 0-7
    rpm data will have the bytes from 8 to 167
    crc will have the bytes from 168 to 169
    '''

    crc_correction = []
    byte_start = datablock_num*dbs # set byte start 

    # Unpacking date_time's bytes
    # The entry for date_time is a 64-bit unsigned integer
    #       64 bits = 8 bytes (16*0.125)
    #       using python's struct.unpack, an 8-byte integer is a long long but since this is time, we need the unsigned long long
    #       unsigned long long = Q (https://docs.python.org/3/library/struct.html#format-characters)
    (ticks,) = unpack("Q", data[byte_start:byte_start+8])

    # Calculating date_time from ticks
    # more info here: https://learn.microsoft.com/en-us/dotnet/api/system.datetime.ticks?view=net-7.0
    #   Note: the time is referenced from January 1, 2001
    date_time = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds = ticks/10)

    # Unpacking gyro's bytes
    # Each sample is a 16-bit signed integer, 50 counts / RPM. Axial gyro values
    #       16 bits = 2 bytes (16*0.125)
    #       using python struct.unpack, a 2-byte integer (signed) is referenced as h (https://docs.python.org/3/library/struct.html#format-characters)
    #
    # Additionally, in total, there are 80 'h' since:
    #       every rpm value is 2 bytes = 2 digits
    #       and the field length is 160
    #       Therefore, 160/2 = 80. This means that we have 80 signed/unsigned integers for this field
    rpm_tuple = unpack('h'*80, data[byte_start+8:byte_start+168])
    
    # Unpacking the bytes for crc
    # this was left as is because it is not part of the actual data to be used for analysis.
    # crc_correction.append(data[byte_start+168:byte_start+170])

    return date_time,rpm_tuple

def import_gyro_bin(file,dbs=170):

    '''
    Created         2022/11/27 scott.hess@ucalgary.ca\n
    Last Modified   2023/01/27 scott.hess@ucalgary.ca\n
    Couple of notes:\n
    Function to import high frequency gyro rpm data in binary format.\n
    Inputs: file - file with path\n
            dbs - datablock size as defined in binary data information sheet\n
    Outputs:\n
            df - pandas dataframe with DateTime/n
            fs_gyro - average sampling frequency

    '''

    with open(file,'rb') as f:
        gyro_data = f.read()

    print('File: ',file)

    # Loop to read data blocks and append to pandas dataframe

    error_flag = 0                  # flag to stop if equal to 1 to account for the case of partial datablock at the end of file
    clock_samples = 0               # counter for number of clock samples read. Dataframe will have clock_sample*24 rows
    Date_Time = []
    rpm = []

    while error_flag ==0:

        try:
            date_time, rpm_data = svgyro_bin2list(gyro_data,clock_samples,dbs)
            if clock_samples == 0:
                Date_Time.append(date_time)   
            else:
                temp_ind = (clock_samples-1)*len(rpm_data)
                tdelta = date_time - Date_Time[temp_ind]
                t_interp = Date_Time[temp_ind]+np.arange(1,len(rpm_data)+1)*tdelta/len(rpm_data)
                Date_Time.extend(t_interp)
            rpm_list = [rpm for rpm in rpm_data]
            rpm.extend(rpm_list)
            clock_samples += 1 # increment to read the next datablock
        except:
            # print('error')
            error_flag = 1

        if len(rpm)%1000000==0:
            print('%i samples read...'%len(rpm))

    del rpm[len(rpm)-len(rpm_list)+1:]
    print('Finished reading %i samples'%len(rpm))
    print('Start time: ',Date_Time[0])
    print('End time: ',Date_Time[-1])
    print('Writing to dataframe....')
    df = pd.DataFrame({'DateTime':Date_Time,'RPM':np.array(rpm)/50})
    df['DateTime'] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d %H:%M:%S.%f")
    total_sec = (df['DateTime'].iloc[-1] - df['DateTime'].iloc[0]).total_seconds()
    fs_mean = df.shape[0]/total_sec
    print('Average gyro sampling frequency: %i Hz'%fs_mean)


    return df, fs_mean

def svaccel_bin2list(data,datablock_num=0,dbs=154):
    ''' 
    Created         2022/12/05 ivanpaul.serrano@ucalgary.ca\n
    Last Modified   2022/12/14 scott.hess@ucalgary.ca\n
    Couple of notes:\n
    'data' is a string of bytes that was used to store the information from the .bin file. \n
    the term 'byte string' for this script is when we are slicing 'data' into chunks.\n
    therefore:\n
    byte string for date_time = data[:8]
    byte string for accelerometer data = data[8:152]
    bte string for crc = data[152:154]
    for python slicing, the format is [m:n]. With this syntax, we are including all values starting from m up to but not including n
    therefore, date_time will have the bytes from 0-7
    accelerometer data will have the bytes from 8 to 151
    crc will have the bytes from 152 to 153
    '''

    x_list = []
    y_list = []
    z_list = []
    crc_correction = []
    
    byte_start = datablock_num*dbs # set byte start to read from desired point of file

    # Unpacking date_time's bytes
    # The entry for date_time is a 64-bit unsigned integer
    #       64 bits = 8 bytes (16*0.125)
    #       using python's struct.unpack, an 8-byte integer is a long long but since this is time, we need the unsigned long long
    #       unsigned long long = Q (https://docs.python.org/3/library/struct.html#format-characters)
    (ticks,) = unpack("Q", data[byte_start:byte_start+8])

    # Calculating date_time from ticks
    # more info here: https://learn.microsoft.com/en-us/dotnet/api/system.datetime.ticks?view=net-7.0
    #   Note: the time is referenced from January 1, 2001
    date_time = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds = ticks/10)

    # Unpacking accelerometer's bytes
    # Each entries for X, Y, Z axis for accelerometer is a 16-bit signed integer
    #       16 bits = 2 bytes (16*0.125)
    #       using python struct.unpack, a 2-byte integer (signed) is referenced as h (https://docs.python.org/3/library/struct.html#format-characters)
    #
    # Additionally, in total, there are 72 'h' since:
    #       every accelerometer value is 2 bytes = 2 digits
    #       and the field length is 144
    #       Therefore, 144/2 = 72. This means that we have 72 signed/unsigned integers for this field
    #       and since that 72 integers are grouped into 3, we'll have 24 (72/3) entries of x, y, z for each of the data chunks in this .bin file
    accel_data = unpack("h"*72, data[byte_start+8:byte_start+152])

    # Unpacking tuple to x, y, z lists
    x_list = [accel_data[x] for x in list(range(0,len(accel_data),3))]
    y_list = [accel_data[y] for y in list(range(1,len(accel_data),3))]
    z_list = [accel_data[z] for z in list(range(2,len(accel_data),3))]

    # Unpacking the bytes for crc
    # this was left as is because it is not part of the actual data to be used for analysis.
    # crc_correction.append(data[byte_start+152:byte_start+154])

    return date_time,x_list,y_list,z_list


def import_accel_bin(file,dbs=154):

    '''
    Created         2023/01/27 scott.hess@ucalgary.ca\n
    Last Modified   2022/11/27 scott.hess@ucalgary.ca\n
    Couple of notes:\n
    Function to import high frequency gyro rpm data in binary format.
    Inputs: file - file with path
            dbs - datablock size as defined in binary data information sheet
    Calls 

    '''

    with open(file,'rb') as f:
        accel_data = f.read()

    print('File: ',file)

    # Loop to read data blocks and append to list

    error_flag = 0                  # flag to stop if equal to 1 to account for the case of partial datablock at the end of file
    clock_samples = 0               # counter for number of clock samples read. Dataframe will have clock_sample*24 rows
    Date_Time = []
    x = []
    y = []
    z = []

    while error_flag ==0:

        try:
            date_time,x_list,y_list,z_list = svaccel_bin2list(accel_data,clock_samples)
            if clock_samples == 0:
                Date_Time.append(date_time)   
            else:
                temp_ind = (clock_samples-1)*len(x_list)
                tdelta = date_time - Date_Time[temp_ind]
                t_interp = Date_Time[temp_ind]+np.arange(1,len(x_list)+1)*tdelta/len(x_list)
                Date_Time.extend(t_interp)
            x.extend(x_list)
            y.extend(y_list)
            z.extend(z_list)
            clock_samples += 1 # increment to read the next datablock
        except:
            # print('error')
            error_flag = 1

        if len(x)%(len(x_list)*1e4)==0:
            print('%i samples read...'%len(x))

        # if len(x)>len(x_list)*1e6:
        #     error_flag=1

    del x[len(x)-len(x_list)+1:]
    del y[len(y)-len(y_list)+1:]
    del z[len(z)-len(z_list)+1:]
    print('Finished reading %i samples'%len(x))
    print('Start time: ',Date_Time[0])
    print('End time: ',Date_Time[-1])
    print('Writing to dataframe....')

    df = pd.DataFrame({'DateTime':Date_Time,'sv_axial':np.array(x)/100,'sv_tang':np.array(y)/100,'sv_radial':np.array(z)/100})
    df['DateTime'] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d %H:%M:%S.%f")

    total_sec = (df['DateTime'].iloc[-1] - df['DateTime'].iloc[0]).total_seconds()
    fs_accel = df.shape[0]/total_sec
    print('Average Accel sampling frequency: %i Hz'%fs_accel)

    return df, fs_accel

def attr_1hz(df,keys,agg_fns=['min','max','mean','std'],datetime = 'DateTime'):
    '''
    Created 2022/08/01 by Scott Hess scott.hess@ucalgary.ca
    Last Modified 2023/03/02 by Scott Hess scott.hess@ucalgary.ca\n\n

    Function to resample high frequency data attributes at 1Hz\n
    Consider using detrended signal using df[arg]=signal.detrend(df[arg])\n\n

    inputs. \n
    df - pandas dataframe of the high frequency data \n
    keys - list of columns from data to calculate attributes\n
    agg_fns - list of aggregate functions to apply to keys [min,max,std,sum,mean,median,mode,var,mad,skew,sem]\n
    datetime - string column name for the datetime column in the hf dataframe\n\n
    
    outputs\n
    df_out - pandas dataframe with the envelope attributes

    Can also create custom functions. Example
    def peak(series):
        return np.max(abs(series))
    Need to figure out how to add into agg function

    '''
     
    dict_col ={}
    for key in keys:
        dict_col[key] = agg_fns
        
    df_out = df.groupby(df[datetime].dt.floor('S')).agg(dict_col)
    df_out.columns = df_out.columns.map('_'.join).str.strip('_')
    df_out.reset_index(inplace=True)
        
    return df_out

def import_csv(file,datetime='DateTime',col_merge=True):
    '''
    Created 2023/01/26 by Scott Hess scott.hess@ucalgary.ca\n
    Modified 2023/02/05 by Scott Hess scott.hess@ucalgary.ca\n\n
    Function to import 1hz drilling data that will merge columns in the format of 'YYYY/MM/DD' and 'HH:MM:SS'\n\n
    inputs\n
    file - drilling data in csv format. 1 second sampling\n
    datetime - str containing name of datetime column in import file csv
    col_merge - bool, set to false if columns do not need merged to create 
    output\n
    df - pandas dataframe.

    Updates:
    2023.04.21 Scott Hess added col_merge flag for when columns do not need to be combined. Added datetime argument to set existing name for datetime column in csv

    '''

    df = pd.read_csv(file)
    if col_merge==True:
        df['DateTime'] = df['YYYY/MM/DD'] + ' ' + df['HH:MM:SS']
        df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y/%m/%d %H:%M:%S")
        df.drop(columns=['YYYY/MM/DD','HH:MM:SS'], inplace=True)
    else:
        df['DateTime'] = pd.to_datetime(df[datetime], format="%Y/%m/%d %H:%M:%S")
        if datetime != 'DateTime':
            df.drop(columns=[datetime], inplace=True)

    return df

    return df

def bulk_shift_calc(df_1,df_2,left_on,right_on,start=601,win=600,datetime='DateTime',plot=False):
    '''
    Created 2023/01/26 by Scott Hess scott.hess@ucalgary.ca\n
    Modified 2023/02/05 by Scott Hess scott.hess@ucalgary.ca\n\n
    Function to calculate bulk shift based on max correlation and plot correlation coefficient vs shift\n\n
    inputs\n
    df_1 - high frequency dataframe. Data will be shifted\n
    df_2 - drilling dataframe. Data will not be shifted. Considered true time\n
    left_on - high frequency dataframe column for correlation\n
    right_on - drilling dataframe column for correlation\n
    start - initial guess for shift in seconds\n
    win - window around initial start shift to calculate correlations\n
    plot - optional plot of calculated correlation coefficients\n\n
    output\n
    shift - best time shift in seconds to apply to high frequency data.

    '''
    # Combine dataframes
    df_corr = pd.merge(df_1,df_2,how='outer',on=datetime)

    if win > start:
        start = win + 1

    # create list of timeshifts to test and empty list for coefficients
    timeshift = list(np.arange(start-win,start+win,1))
    coeff = []

    for shift in timeshift:
        # print(df_corr.shift(-shift)[left_on].fillna(0).shape)
        # print(df_corr[right_on].fillna(0).shape)
        cc_coef = np.correlate(np.squeeze(df_corr.shift(-shift)[left_on].fillna(0)),df_corr[right_on].fillna(0))
        coeff = np.append(coeff,np.cumsum(cc_coef))

    shift = timeshift[coeff.argmax()]
    # optional plot to show
    if plot == True:
    
        plt.plot(timeshift,coeff)
        ylim = plt.gca().get_ylim()
        plt.vlines(shift,ylim[0],ylim[1])
        plt.ylabel('Cross-Corr Coef'), plt.xlabel('shift (seconds)')
        plt.gcf().suptitle('Best shift for ' + left_on + ' to match ' + right_on)
        plt.gcf().set_size_inches(5,3)
        plt.show()

    return shift

def apply_shift(df,shift,datetime='DateTime'):
    '''
    Created 2023/01/26 by Scott Hess scott.hess@ucalgary.ca\n
    Modified 2023/02/27 by Scott Hess scott.hess@ucalgary.ca\n\n
    Function to apply bulk shift to dataframe datetime and save original timestamps\n\n
    inputs\n
    df - Dataframe to be shift. This will generally be downhole memory data\n
    shift - Shift in seconds to be applied. \n
    datetime - column name of panda dataframe time series\n
    output\n
    returns modified input dataframe.
    '''
    if datetime+'_raw' in df.keys():
        pass
    else:
        df[datetime+'_raw'] = df[datetime]

    df[datetime] = df[datetime]-pd.to_timedelta(np.round(shift),'S')
    print('%i second shift applied'%shift)
    # print('%i second shift applied to %s'%(shift,df.name))
    print('Original time saved to',datetime+'_raw')
    print(df.keys())

def reset_shift(df,datetime='DateTime'):

    '''
    Created 2023/01/26 by Scott Hess scott.hess@ucalgary.ca\n
    Modified 2023/02/27 by Scott Hess scott.hess@ucalgary.ca\n\n
    Function to apply reset datetime columnof dataframe to original timestamps\n\n
    inputs\n
    df - Dataframe to be shift. This will generally be downhole memory data\n
    datetime - column name of panda dataframe time series\n
    output\n
    returns None. Modifies input dataframe.
    '''

    if datetime+'_raw' in df.keys():
        df[datetime] = df[datetime+'_raw']
        df.drop(columns='DateTime_raw',inplace=True)
        print('Original datetime reset')
        print(df.keys())
    else:
        print('No shift applied')

def merge_1hz(df_1,df_2,win=[],datetime='DateTime'):
    try :
        df_out = df_1[df_1[datetime].between(win[0],win[1])].merge(df_2[df_2[datetime].between(win[0],win[1])],how='left', on=datetime).copy()
    except:
        df_out = df_1.merge(df_2,how='left', on=datetime)
    
    return df_out

def csv_export(df,pathfilename,win=[],datetime='DateTime'):
    try:
        df[df[datetime].between(win[0],win[1])].to_csv(pathfilename)
    except:
        df.to_csv(pathfilename)

def drift_corr(df,d_t,d_v,datetime='DateTime',mode1hz=False,plot=False):

    '''
    Created 2022/02/28 by Scott Hess scott.hess@ucalgary.ca\n
    Modified 2023/03/14 by Scott Hess scott.hess@ucalgary.ca\n\n
    Function to apply drift correction to datetime column using linear regression fit for drift values\n\n
    inputs\n
    df - Dataframe to be drift corrected. This will generally be downhole memory data\n
    d_t - List of timestamps starting with zero drift timestamp followed by additional tie point timestamps\n
    d_v - List of drift values in seconds starting with zero followed by drift values for tie points\n
    datetime - column name of panda dataframe time series\n
    mode1hz - Simply turns on or off rounding to second for delta times\n
    plot - option to plot linear regression fit\n
    output\n
    returns None. Modifies input dataframe.
    '''

    # convert times to delta time relative to drift zero time in seconds
    d_ts = [((t-d_t[0]).total_seconds()) for t in d_t]

    # drift estimation
    linreg = np.polyfit(d_ts,d_v,1)
    x0 = (df[datetime].iloc[0]-d_t[0]).total_seconds()
    xmax = (df[datetime].iloc[-1]-d_t[0]).total_seconds()
    x = np.linspace(x0,xmax,df.shape[0])
    y = linreg[1]+linreg[0]*x

    if datetime+'_raw' in df.keys():
        pass
    else:
        df[datetime+'_raw'] = df[datetime]
    if mode1hz==True:
        df[datetime] = df[datetime]-pd.to_timedelta(np.round(y),'S')
    elif mode1hz==False:
        df[datetime] = df[datetime]-pd.to_timedelta(y)
    # 
    df[datetime] = df[datetime]-pd.to_timedelta(y)
    if plot == True:

        plt.plot(d_ts,d_v,'o')
        plt.plot(x,y)
        plt.xlabel('Time Relative to zero drift time (s)')
        plt.ylabel('Clock Drift (s)')
        plt.show()


def stand_detect(df, curve_names=['Hole Depth (feet)','Bit Depth (feet)','Block Height (feet)','Rate Of Penetration (ft_per_hr)'],threshold=0.5):
     
    '''
    Created Unknown by Ivan\n
    Modified 2023/03/14 by Scott Hess scott.hess@ucalgary.ca\n\n
    Function to auto-calculate stands for 1hz drilling data\n\n
    inputs\n
    df - pandas Dataframe, drilling data\n
    curve_names - [str] column names for hole depth, bit depth, block height, and rate of penetration. In that order.
    threshold - float, rate of penetration threshold\n
    
    output\n
    updated_dataframe_with_stands - pandas dataframe, input dataframe including stands number in 'Stand' column.

    Updates
    2023.04.21 added curve_names arg for flexible column naming
    '''

    if 'Stand' in df.keys():
        df.drop(columns='Stand',inplace=True)

    stand = False
    stand_idx = 0
    df_stands_dict = df.to_dict('records')
    updated_dataframe = pd.DataFrame()
    concat_dataframe_stands = pd.DataFrame()
    dict_stands = {}
    stands_dict = []
    updated_stands_dict = []
    block_height_list = []
    rop_list = []

    for index in df_stands_dict:
        block_height_list.append(index[curve_names[2]])
        rop_list.append(index[curve_names[3]])

    max_block_height = max(block_height_list)
    min_block_height = min(block_height_list)

    for row in df_stands_dict: 
        if stand is False:
            if row[curve_names[0]] - row[curve_names[1]] < 50:
                if max_block_height - 10 <= row[curve_names[2]] <= max_block_height + 10 and row[curve_names[3]] < threshold:
                    stand = True
                    stand_idx += 1
                    stands_dict.append(stand_idx)
                    updated_stands_dict.append(row)
                    
        else:
            if min_block_height - 10 <= row[curve_names[2]] <= min_block_height + 10 and row[curve_names[3]] < threshold:
                stand = False    

            else:
                stands_dict.append(stand_idx)
                updated_stands_dict.append(row)

    dict_stands['Stand'] = stands_dict
    updated_dataframe = pd.concat([updated_dataframe, pd.DataFrame.from_dict(dict_stands)])
    concat_dataframe_stands = pd.concat([concat_dataframe_stands, pd.DataFrame(updated_stands_dict)])
    concat_dataframe_stands = concat_dataframe_stands.loc[:, ~concat_dataframe_stands.columns.str.contains('^Unnamed')]
    updated_dataframe_with_stands = pd.concat([updated_dataframe, concat_dataframe_stands], axis = 1)
    updated_dataframe_with_stands['Stand'] = updated_dataframe_with_stands['Stand'].fillna(0)

    return updated_dataframe_with_stands