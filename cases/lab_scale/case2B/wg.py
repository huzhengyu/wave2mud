#!/usr/bin/python
# HZY NEW
import os 

pathname = os.path.abspath('.')
savePath = os.path.join(pathname,'WG')
if not os.path.isdir(savePath):
    os.makedirs(savePath)

postPath = os.path.join(pathname,'postProcessing/gaugesVOF')
if not os.path.isdir(postPath):
    postPath = 'gaugesVOF/fluid'

# List of time dirs in order
a = sorted(os.listdir(postPath), key=float)

# Get number of sensors
dir1 = os.path.join(pathname,postPath,a[int(len(a)/2.0)])
b = os.listdir(dir1)
nSens = 0
index = []
for i in range(len(b)):
    test1 = b[i].find('Wave') + 1
    test2 = b[i].find('alpha') + 1
    if test1 and test2:
        index.append(i)
        nSens += 1

first = True
b = sorted(b, key=str)
Gauge_x = 'Time '
for i in range(nSens):
    # Create files to write
    fileName = b[index[i]][0:b[index[i]].find('_')]
    fileW = open(os.path.join(savePath,fileName), 'w')
    print(str(i+1) + ' of ' + '%i' % nSens + ' gauges.')

    # Read files time by time
    for j in range(len(a)):
        directory = os.path.join(pathname,postPath,a[j])
        try:
            fileR = open(os.path.join(directory,b[index[i]]), 'r')
        except:
            print('WARNING - File not present: ' + os.path.join(directory,b[index[i]]))
        else:
            data = fileR.read()
            fileR.close()
            data = data.split('\n')
                      
            if first: # First time step
                coord = j
                first = False
            
            # x = []
            # y = []
            z = []
            alpha = []
    
            # x y z alpha1 calculation
            for k in range(len(data)-1):
                line = data[k]
                line = line.split('\t') 
                # x = float(line[0]) # y = float(line[1]) 
                # z = float(line[2]) # pres = float(line[3])
                
                z.append(float(line[2]))
                if b[index[i]][:3] == 'mud':
                    alpha.append(1.0-float(line[3]))
                else:
                    alpha.append(float(line[3]))
                
                # if j == coord: # First time step
                    # Create coordinate files
                    # fileWXYZ = open(os.path.join(savePath,fileName + '.xy'), 'w')
                    # fileWXYZ.write( line[0] + line[1] )
                    # fileWXYZ.close()
                    # Gauge_x.append(line[0]+' ')
            if j == coord:
                Gauge_x+=line[0]
            # Integrate in Z
            wLevel = z[0]
            for k in range(len(z)-1):
                wLevel = wLevel + (alpha[k]+alpha[k+1])*(z[k+1]-z[k])/2
    
            # Write to file
            time = a[j]
            fileW.write(time + ' ' + '%.6f' % wLevel + '\n')

    fileW.close()
print('Merging...')
 
for i in range(nSens):
        # Create files to write
        fileName = b[index[i]][0:b[index[i]].find('_')]
        fileR = open(os.path.join(savePath,fileName), 'r')
        data = fileR.read()
        data = data.split('\n')
        if i == 0:
            WG = data
        else:
            for j in range(len(data)-1):
                WG[j] = WG[j]+' '+data[j].split()[1]
        fileR.close()

fileW = open(os.path.join(savePath,'WG'), 'w')
fileW.write(Gauge_x+'\n')
for j in range(len(data)):
    fileW.write(WG[j] + '\n')
fileW.close()
print('Done')
