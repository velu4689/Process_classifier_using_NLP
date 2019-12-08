# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:43:19 2018

@author: velmurugan.m
"""

import pandas as pd
import numpy as np

###############################################################################            
def analyzeGUID(p2GUID, p3GUID, p4TarGUID, p4HostGUID):

    #### Get the Entry Count for specific GUIDs from Part-2 & Part-3 
    arrTarCntList = p4TarGUID[p2GUID]
    arrHosCntList = p4HostGUID[p2GUID]     
    
###############################################################################            
def part2Analysis():
    Loc2 = r'D:\ATT\newData\p2\oem_metrics.csv'    
    oemMetricDF = pd.read_csv(Loc2)    
    
    #### Get the Labels for Part 3 of the Data Series 
    oemMetricColLabels = oemMetricDF.columns.values    
    
    #### Get the GUID Unique values for Part 2 of the Data Series 
    oemMetricTarGUIDUniq = oemMetricDF.target_guid.unique()
    
    #### Get the GUID values for Part 2 of the Data Series
    oemMetricTarGUID = oemMetricDF.target_guid
    
    
        
    return oemMetricTarGUID

###############################################################################       
def part3Analysis():
    Loc3 = r'D:\ATT\newData\p3\oem_metrics_details.csv'    
    oemMetricDetailsDF = pd.read_csv(Loc3)        
    
    #### Get the Labels for Part 3 of the Data Series
    oemMetricDetColLabels = oemMetricDetailsDF.columns.values
    
    #### Get the GUID Unique values for Part 3 of the Data Series
    oemMetricDetTarGUIDUniq = oemMetricDetailsDF.target_guid.unique()
    
    #### Get the GUID values for Part 3 of the Data Series
    oemMetricDetTarGUID = oemMetricDetailsDF.target_guid     
    
    return oemMetricDetTarGUID

###############################################################################                
def part4Analysis():
    Loc4 = r'D:\ATT\newData\p4\oem_target_details.csv'           
    oemTargetDetailsDF = pd.read_csv(Loc4)
    
    #### Get the Target Labels for Part 4 of the Data Series
    oemTargetDetColLabels = oemTargetDetailsDF.columns.values
    oemTargetDetTarGUIDUniq = oemTargetDetailsDF.target_guid.unique()
    oemTargetDetHostGUIDUniq = oemTargetDetailsDF.host_target_guid.unique()
    
    #### Get the Target GUID Value Counts for Part 4 of the Data Series 
    oemTargetDetTarGUIDValCounts = oemTargetDetailsDF.target_guid.value_counts()    
    oemTargetDetHostGUIDValCounts = oemTargetDetailsDF.host_target_guid.value_counts()    
    
    #### Get the Target + Host GUID Columns for Part 4 of the Data Series
    oemTargetDetTarGUID = oemTargetDetailsDF.target_guid
    oemTargetDetHostGUID = oemTargetDetailsDF.host_target_guid
    
    return (oemTargetDetTarGUIDValCounts, oemTargetDetHostGUIDValCounts)
    
###############################################################################
def main():
    p2GUID = part2Analysis()
    #p3GUID = part3Analysis()
    #p4TarGUIDValCnts, p4HostGUIDValCnts = part4Analysis()
    
    #analyzeGUID( p2GUID, p3GUID, p4TarGUIDValCnts, p4HostGUIDValCnts )

if __name__ == "__main__":
    main()