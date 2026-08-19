#!/usr/bin/env python3

# Tools to read/write NICOLE models and profiles

def check_types():
    import struct
    import sys

# Check size of default types. Need 32-bit integers, 64-bit integers and floats
# 32-bit integers
    if struct.calcsize('<i') == 4:
        int4f='i'
    elif struct.calcsize('<l') == 4:
        int4f='l'
    elif struct.calcsize('<q') == 4:
        int4f='q'
    else:
        print('This architecture has a non-standard integer size')
        print('which is not supported in this version. Please, contact the')
        print('author to get support for this platform.')
        sys.exit(1)
# 64-bit integers
    if struct.calcsize('<i') == 8:
        intf='i'
    elif struct.calcsize('<l') == 8:
        intf='l'
    elif struct.calcsize('<q') == 8:
        intf='q'
    else:
        print('This architecture has a non-standard integer size')
        print('which is not supported in this version. Please, contact the')
        print('author to get support for this platform.')
        sys.exit(1)
# 64-bit floats
    if struct.calcsize('<f') == 8:
        flf='f'
    elif struct.calcsize('<d') == 8:
        flf='d'
    else:
        print('This architecture has a non-standard float size')
        print('which is not supported in this version. Please, contact the')
        print('author to get support for this platform.')
        sys.exit(1)
    return [int4f,intf,flf]
    

def check_model (filename):
# Check the input model and determine what kind of file it is
# Return nx, ny and nz
    import re
    import os
    import struct
    import sys


    [int4f,intf,flf]=check_types()
    try:
        f=open(filename,'r')
    except:
        print('Could not find model file:',filename)
        return ['inexistent',-1,-1,-1]
    filetype='unknown'
    # Check if it's nicole's binary format
    f.close()
    f=open(filename,'rb')
    header=f.read(16+4+4+8)
    [string,ny,nz]=struct.unpack('<16s'+intf+intf,header)
    string=string[0:11] # Remove padding zeros at the end
    if string == b'nicole1.6bm': # Older format version
        filetype='nicole1.6'
        nx=1
        [string,ny,nz]=struct.unpack('<16s'+intf+intf,header)
        f.close()
        filesize=os.path.getsize(filename)
        if filesize != (13*nz+3)*(nx*ny+1)*8:
            print('Incorrect size of model file:',filename)
            print('The file is probably corrupted. Proceeding anyway...')
#            sys.exit(1)
        return [filetype,nx,ny,nz]
    if string == b'nicole2.3bm': # Old format version
        filetype='nicole2.3'
        [string,nx,ny,nz]=struct.unpack('<16s'+int4f+int4f+intf,header)
        f.close()
        filesize=os.path.getsize(filename)
        if filesize != (17*nz+3)*(nx*ny+1)*8:
            print('Incorrect size of model file:',filename)
            print('The file is probably corrupted. Proceeding anyway...')
#            sys.exit(1)
        return [filetype,nx,ny,nz]
    if string == b'nicole2.6bm': # Current format version
        filetype='nicole2.6'
        [string,nx,ny,nz]=struct.unpack('<16s'+int4f+int4f+intf,header)
        f.close()
        filesize=os.path.getsize(filename)
        if filesize != (22*nz+3+8+92)*(nx*ny+1)*8:
            print('Incorrect size of model file:',filename)
            print('The file is probably corrupted. Proceeding anyway...')
#            sys.exit(1)
        return [filetype,nx,ny,nz]
    f.close()
    # Check if it's ASCII    
    f=open(filename,'r')
    line='---'
    while not re.search('(?i)format version',line) and line != '':
        line=f.readline()
    if re.search('(?i)format version',line):
        filetype='ascii'
        filetype=filetype+line.split()[len(line.split())-1]
        if filetype != 'ascii1.0' and filetype != 'ascii2.3':
            print('Model file:',filename)
            print('seems to be an ASCII file of unsupported format')
            print(line)
            sys.exit(1)
        # Count number of model lines
        string=f.read()
        string=re.sub('!.*\n','\n',string)
        string=re.sub('(?i).*format version.*\n','\n',string)
        lines=string.rsplit('\n')
        for l in range(lines.count('')):
            lines.remove('')
        nz=len(lines)-1
        #
        f.close()
        nx=1
        ny=1
        return [filetype,nx,ny,nz]
    # Check if it's an IDL save file
    f=open(filename,'rb')
    stri=struct.unpack('10s',f.read(10))
    stri=stri[0]
    bytes_list=list()
    for i in range(len(stri)): bytes_list.append(stri[i] if isinstance(stri[i], int) else ord(stri[i]))
    f.close()
    if bytes_list == [83, 82, 0, 4, 0, 0, 0, 10, 0, 0]:
        filetype='idl'
        try:
            import idlsave
        except Exception:
            print('Error. Model file is an IDL save file:',filename)
            print('Python modules IDLSave and Numpy are needed to read this kind of files')
            print("The modules don't appear to be installed in your system")
            sys.exit(1)
        idl=idlsave.read(filename,verbose=0)
        shape=idl.t.shape
        ls=list(shape)
        while (len(ls) < 3):
#           ls.insert(0,1)
           ls.append(1)
        idl.t=idl.t.reshape(ls)
        try:
            nz=idl.nz
        except:
            nz=idl.t.shape[0]
        try:
            ny=idl.ny
        except:
            ny=idl.t.shape[1]
        try:
            nx=idl.nx
        except:
            nx=idl.t.shape[2]
        return [filetype,nx,ny,nz]
#
    print('Unknown model file type:'+filename)
    sys.exit(1)

def check_prof (filename):
# Check the input profile and determine what kind of file it is
# Return nx, ny and nlambda
    import re
    import os
    import string
    import struct
    import sys

    [int4f,intf,flf]=check_types()
    try:
        f=open(filename,'r')
    except:
        print('Could not find profile file:',filename)
        return ['inexistent',-1,-1,-1]
    filetype='unknown'
    # First check if it's ASCII    
    readstr=f.read(1000000)
    f.close()
    nprintable=0
    for char in readstr: 
        if char in(string.printable) or char in(string.whitespace): 
            nprintable=nprintable+1
    if len(readstr) == 0:
        print('Empty file!')
        print('filename:',filename)
        sys.exit(1)
    if float(nprintable)/float(len(readstr)) > .95:
        filetype='ascii'
        # Count number of wavelengths
        f=open(filename,'r')
        string_content=f.read()
        string_content=re.sub('!.*\n','\n',string_content)
        lines=string_content.rsplit('\n')
        for l in range(lines.count('')):
            lines.remove('')
        nlam=len(lines)
        #
        f.close()
        nx=1
        ny=1
        return [filetype,nx,ny,nlam]
    # Check if it's nicole's binary format
    f=open(filename,'rb')
    header=f.read(16+8+8)
    [readstr,ny,nlam]=struct.unpack('<16s'+intf+intf,header)
    readstr=readstr[0:11]
    if readstr == b'nicole1.6bp': # Older format version
        filetype='nicole1.6'
        nx=1
        [readstr,ny,nlam]=struct.unpack('<16s'+intf+intf,header)
        f.close()
        filesize=os.path.getsize(filename)
        if filesize != (4*nlam)*(nx*ny+1)*8:
            print('Incorrect size of profile file:',filename)
            print('The file is probably corrupted. Proceeding anyway...')
#            sys.exit(1)
        return [filetype,nx,ny,nlam]
    if readstr == b'nicole2.3bp': # Current format version
        filetype='nicole2.3'
        [readstr,nx,ny,nlam]=struct.unpack('<16s'+int4f+int4f+intf,header)
        f.close()
        filesize=os.path.getsize(filename)
        if filesize != (4*nlam)*(nx*ny+1)*8:
            print('Incorrect size of profile file:',filename)
            print('The file is probably corrupted. Proceeding anyway...')
#            sys.exit(1)
        return [filetype,nx,ny,nlam]
    f.close()
    # Check if it's an IDL save file
    f=open(filename,'rb')
    stri=struct.unpack('10s',f.read(10))
    stri=stri[0]
    bytes_list=list()
    for i in range(len(stri)): bytes_list.append(stri[i] if isinstance(stri[i], int) else ord(stri[i]))
    f.close()
    if bytes_list == [83, 82, 0, 4, 0, 0, 0, 10, 0, 0]:
        filetype='idl'
        try:
            import idlsave
        except Exception:
            print('Error. Profile file is an IDL save file:',filename)
            print('Python modules IDLSave and Numpy are needed to read this kind of files')
            print("The modules don't appear to be installed in your system")
            sys.exit(1)
        idl=idlsave.read(filename,verbose=0)
        shape=idl.stki.shape
        ls=list(shape)
        while (len(ls) < 3):
#            ls.insert(0,1)
            ls.append(1)
        idl.stki=idl.stki.reshape(ls)
        try:
            nlam=idl.nlam
        except:
            nlam=idl.stki.shape[0]
        try:
            ny=idl.ny
        except:
            ny=idl.stki.shape[1]
        try:
            nx=idl.nx
        except:
            nx=idl.stki.shape[2]
        return [filetype,nx,ny,nlam]
#
    print('Unknown profile file type:'+filename)
    sys.exit(1)

def read_prof(filename, filetype, nx, ny, nlam, ix, iy, sequential=0):
# Read a particular profile from a profile file 
# Note that ix,iy=0 represents the first profile
# Use the sequential keyword to specify that this file is going to be
# read sequentially rather than accessing specific records. In that
# case, the values of ix and iy are irrelevant. The first record
# needs to be read with sequential=0 and ix,iy = 0
    import struct
    import re
    import sys
    global idl, irec, f # Save values between calls

    [int4f,intf,flf]=check_types()
# Read a particular profile from a profile file 
# Note that ix, iy=0 represents the first profile
    if ix < 0 or iy < 0: 
        print('Error in call to read_prof')
        sys.exit(1)
    if filetype == 'ascii':
        if ix != 0 and iy != 0:
            print('Error in read_prof')
            print('Attempt to read ix,iy different from 0 from ascii file:',filename)
            print('Requested ix,iy:',ix,iy)
            sys.exit(1)
        f=open(filename,'r')
        readstr=f.read()
        readstr=re.sub('!.*\n','\n',readstr)
        lines=readstr.rsplit('\n')
        for l in range(lines.count('')):
            lines.remove('')        
        f.close()
        if len(lines) != nlam:
            print('Error reading profile file:',filename)
            print('Incorrect number of wavelengths')
            sys.exit(1)
        data=list()
        for l in lines:
            row=l.split()    
            for istk in range(4):
                data.append(row[istk+1])
        for i in range(len(data)): data[i]=float(data[i])
        return data
    elif filetype[0:6] == 'nicole':
        if (sequential == 0):
            irec=iy+ix*ny
        else:
            irec=irec+1
        sizerec=4*nlam # Floats (multiply by 8 to convert to bytes)
        if (sequential == 0):
            f=open(filename,'rb')
            f.seek(sizerec*8*(irec+1)) # Skip header and previous records
        data=struct.unpack('<'+str(sizerec)+flf,f.read(sizerec*8))
        data=list(data)
        return data
    elif filetype == 'idl':
        import idlsave
        try: 
            a=idl.stki[0,0,0]
        except:
            idl=idlsave.read(filename,verbose=0)
            idl.stki=idl.stki.reshape(nlam,ny,nx)
            idl.stkq=idl.stkq.reshape(nlam,ny,nx)
            idl.stku=idl.stku.reshape(nlam,ny,nx)
            idl.stkv=idl.stkv.reshape(nlam,ny,nx)
        data=list()
        for ilam in range(nlam): 
            data.append(idl.stki[ilam,iy,ix])
            data.append(idl.stkq[ilam,iy,ix])
            data.append(idl.stku[ilam,iy,ix])
            data.append(idl.stkv[ilam,iy,ix])
        for i in range(len(data)): data[i]=float(data[i])
        return data
    else:
        print('Unknown file type')
        sys.exit(1)

def read_model(filename, filetype, nx, ny, nz, ix, iy, sequential=0):
# Read a particular model from a model file 
# Note that ix,iy=0 represents the first model
# Use the sequential keyword to specify that this file is going to be
# read sequentially rather than accessing specific records. In that
# case, the values of ix and iy are irrelevant. The first record
# needs to be read with sequential=0 and ix,iy = 0
# Model is returned WITHOUT abundances
    import struct
    import re
    import sys
    global idl, irec, f # Save values between calls

    [int4f,intf,flf]=check_types()
    if ix < 0 or iy < 0: 
        print('Error in call to read_model')
        sys.exit(1)
    if filetype[0:5] == 'ascii':
        if ix != 0 or iy != 0:
            print('Error in read_model')
            print('Attempt to read ix,iy different from 0 from ascii file:',filename)
            print('Requested ix,iy:',ix,iy)
            sys.exit(1)
        f=open(filename,'r')
        readstr=f.read()
        readstr=re.sub('!.*\n','\n',readstr)
        readstr=re.sub('(?i).*format version.*\n','\n',readstr)
        lines=readstr.rsplit('\n')
        for l in range(lines.count('')):
            lines.remove('')        
        f.close()
        l=lines[0].split()
        if len(lines)-1 != nz:
            print('Error reading model file:',filename)
            print('Incorrect number of depth points',nz,len(lines))
            sys.exit(1)
        vmac=0.
        stray_frac=0.
        expansion=0.
        if len(l) >= 1: vmac=float(l[0])
        if len(l) >= 2: stray_frac=float(l[1])
        if len(l) >= 3: expansion=float(l[2])       
        data=list()
        for icol in range(22): # Loop in model variables
            for l in lines[1:]: # First line is vmac, stray, expansion
                row=l.split()
                if filetype == 'ascii1.0':
                    if icol == 0:
                        data.append( 0. ) # z
                    elif icol == 1:
                        data.append( row[0] ) # tau
                    elif icol == 2:
                        data.append( row[1] ) # temp
                    elif icol == 3 or icol == 4:
                        data.append( row[2] ) # gas_p or rho
                    elif icol == 5:
                        data.append( row[2] ) # el_p
                    elif icol == 6:
                        data.append( row[5] ) # v_los
                    elif icol == 7:
                        data.append( row[3] ) # v_mic
                    elif icol == 8:
                        data.append( row[4] ) # b_long
                    elif icol == 9:
                        data.append( row[6] ) # b_x
                    elif icol == 10:
                        data.append( row[7] ) # b_y
                    elif icol == 11:
                        data.append( row[6] ) # bl_x
                    elif icol == 12:
                        data.append( row[7] ) # bl_y
                    elif icol == 13:
                        data.append( row[5] ) # by_z
                    elif icol == 14:
                        data.append( 0. ) # v_x
                    elif icol == 15:
                        data.append( 0. ) # v_y
                    elif icol == 16:
                        data.append( row[5] ) # v_z
                    elif icol >= 17:
                        data.append( 0. ) # nH, nH-, nH+, nH2, nH2+
                if filetype == 'ascii2.3':
                    if icol == 0:
                        data.append( row[0] ) # z
                    elif icol == 1:
                        data.append( row[1] ) # tau
                    elif icol == 2:
                        data.append( row[2] ) # temp
                    elif icol == 3:
                        data.append( row[3] ) # gas_p
                    elif icol == 4:
                        data.append( row[4] ) # rho
                    elif icol == 5:
                        data.append( row[5] ) # el_p
                    elif icol == 6:
                        data.append( row[6] ) # v_los
                    elif icol == 7:
                        data.append( row[7] ) # v_mic
                    elif icol == 8:
                        data.append( row[8] ) # b_long
                    elif icol == 9:
                        data.append( row[9] ) # b_x
                    elif icol == 10:
                        data.append( row[10] ) # b_y
                    elif icol == 11:
                        data.append( row[11] ) # bl_x
                    elif icol == 12:
                        data.append( row[12] ) # bl_y
                    elif icol == 13:
                        data.append( row[13] ) # bl_z
                    elif icol == 14:
                        data.append( row[14] ) # vl_x
                    elif icol == 15:
                        data.append( row[15] ) # vl_y
                    elif icol == 16:
                        data.append( row[16] ) # vl_z
                    elif icol >= 17:
                        data.append( 0. ) # nH, nH-, nH+, nH2, nH2+
        data.append(vmac)
        data.append(stray_frac)
        data.append(expansion)
        for i in range(8): data.append(0) # Keep el_p,gas_p,rho,nH,nH-,nH+,nH2,nH2+
        for i in range(len(data)): data[i]=float(data[i])
        return data
    elif filetype == 'nicole1.6':
        irec=iy+ix*ny
        f=open(filename,'rb')
        sizerec=13*nz+3 # Floats (multiply by 8 to convert to bytes)
        f.seek(sizerec*8*(irec+1)) # Skip header and previous records
        data=struct.unpack('<'+str(sizerec)+flf,f.read(sizerec*8))
        data2=list(range(22*nz+3+8))
        data2[0:sizerec]=data[0:sizerec]
        for ind in range(nz):
            data2[6*nz+ind]=data[6*nz+ind] # v_los
        for ind in range(nz):
            data2[8*nz+ind]=data[8*nz+ind] # b_long
        for ind in range(nz):
            data2[9*nz+ind]=data[9*nz+ind] # b_x
        for ind in range(nz):
            data2[10*nz+ind]=data[10*nz+ind] # b_y
        for ind in range(nz):
            data2[11*nz+ind]=data[9*nz+ind] # bl_x=b_x
        for ind in range(nz):
            data2[12*nz+ind]=data[10*nz+ind] # bl_y=b_y
        for ind in range(nz):
            data2[13*nz+ind]=data[8*nz+ind] # bl_z=b_long
        for ind in range(nz):
            data2[14*nz+ind]=0. # vl_x
        for ind in range(nz):
            data2[15*nz+ind]=0. # vl_y
        for ind in range(nz):
            data2[16*nz+ind]=data[6*nz+ind] # bl_y=b_y
        for ivar in range(5):
            for ind in range(nz):
                data2[(17+ivar)*nz+ind]=0. 
        data2[22*nz:22*nz+3]=data[13*nz:13*nz+3] # v_mac, stray_frac, expansion
        for ind in range(8): data2[22*nz+3+ind]=0. # keep
        f.close()
        return data2
    elif filetype == 'nicole2.3':
        if (sequential == 0):
            irec=iy+ix*ny
        else:
            irec=irec+1
        sizerec=17*nz+3 # Floats (multiply by 8 to convert to bytes)
        if (sequential == 0):
            f=open(filename,'rb')
            f.seek(sizerec*8*(irec+1)) # Skip header and previous records
        data=struct.unpack('<'+str(sizerec)+flf,f.read(sizerec*8))
        data2=list()
        for ind in range(17*nz): data2.append(data[ind])
        for ivar in range(5):
            for ind in range(nz):
                data2.append(0.)
        for ind in range(3): data2.append(data[13*nz+ind]) # v_mac, stray_frac, expansion
        for ind in range(8): data2.append(0.) # keep
        for ind in range(len(data2)): data2[ind]=float(data2[ind])
        return data2
    elif filetype == 'nicole2.6':
        if (sequential == 0):
            irec=iy+ix*ny
        else:
            irec=irec+1
        sizerec=22*nz+3+8+92 # Floats (multiply by 8 to convert to bytes)
        if (sequential == 0):
            f=open(filename,'rb')
            f.seek(sizerec*8*(irec+1)) # Skip header and previous records
        data=struct.unpack('<'+str(sizerec)+flf,f.read(sizerec*8))
        data=list(data)
        return data
    elif filetype == 'idl':
        import idlsave
        try: 
            a=idl.z[0,0,0]
        except:
            idl=idlsave.read(filename,verbose=0)
            try:
                idl.z=idl.z.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.tau=idl.tau.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.t=idl.t.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.gas_p=idl.gas_p.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.rho=idl.rho.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.el_p=idl.el_p.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.v_los=idl.v_los.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.v_mic=idl.v_mic.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.b_long=idl.b_los_z.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.b_tx=idl.b_los_x.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.b_ty=idl.b_los_y.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.b_x=idl.b_x.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.b_y=idl.b_y.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.b_z=idl.b_z.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.v_x=idl.v_x.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.v_y=idl.v_y.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.v_z=idl.v_z.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.nH=idl.nH.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.nHminus=idl.nHminus.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.nHplus=idl.nHplus.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.nH2=idl.nH2.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.nH2plus=idl.nH2plus.reshape(nz,ny,nx)
            except: donothing=0
            try:
                idl.v_mac=idl.v_mac.reshape(ny,nx)
            except: donothing=0
            try:
                idl.stray_frac=idl.stray_frac.reshape(ny,nx)
            except: donothing=0
            try:
                idl.expansion=idl.expansion.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_el_p=idl.keep_el_p.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_gas_p=idl.keep_gas_p.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_rho_p=idl.keep_rho_p.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_nH=idl.keep_nH.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_nHminus=idl.keep_nHminus.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_nHplus=idl.keep_nHplus.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_nH2=idl.keep_nH2.reshape(ny,nx)
            except: donothing=0
            try:
                idl.keep_nH2plus=idl.keep_nH2plus.reshape(ny,nx)
            except: donothing=0
            try:
                idl.abundance=idl.abundance.reshape(92,ny,nx)
            except: donothing=0
        data=list()
        try:
            for d in idl.z[:,iy,ix]: data.append(d)
        except: 
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.tau[:,iy,ix]: data.append(d)
        except: 
            for d in range(nz): data.append( 0. )
        for d in idl.t[:,iy,ix]: data.append(d)
        try:
            for d in idl.gas_p[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.rho[:,iy,ix]: data.append(d)
        except: 
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.el_p[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.v_los[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.v_mic[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.b_long[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.b_tx[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.b_ty[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.b_x[:,iy,ix]: data.append(d) 
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.b_y[:,iy,ix]: data.append(d) 
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.b_z[:,iy,ix]: data.append(d) 
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.v_x[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.v_y[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.v_z[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.nH[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.nHminus[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.nHplus[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.nH2[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            for d in idl.nH2plus[:,iy,ix]: data.append(d)
        except:
            for d in range(nz): data.append( 0. )
        try:
            data.append(idl.v_mac[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.stray_frac[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.expansion[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_el_p[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_gas_p[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_rho[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_nH[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_nHminus[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_nHplus[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_nH2[iy,ix])
        except:
            data.append( 0. )
        try:
            data.append(idl.keep_nH2plus[iy,ix])
        except:
            data.append( 0. )
#        try:
#            data.append(idl.abundance[:,iy,ix])
#        except:
#            data.append( 0. )
        for i in range(len(data)): data[i]=float(data[i])
        return data
    else:
        print('Unknown file type')
        sys.exit(1)


def write_ascii_model(
    filepath,
    logtau,
    T,
    v_los_kms,
    b_long_G,
    *,
    v_mic_cms=1.0e5,
    v_mac_cms=0.0,
    stray_light=0.0,
    b_x_G=0.0,
    b_y_G=0.0,
    el_p=None,
    el_p_seed=1.0,
):
    # Write a NICOLE Format-version-1.0 ASCII .model file.
    # MUISCA convention: logtau ascending (top -> bottom).
    # NICOLE convention: rows ordered by descending logtau (deepest -> top).
    # The function reverses internally and converts km/s -> cm/s for v_los.
    # Columns written: log_tau, T(K), El_p(dyn/cm^2), v_mic(cm/s),
    # B_long(G), v_los(cm/s), B_x(G), B_y(G).
    import numpy as np
    from pathlib import Path

    logtau = np.asarray(logtau, dtype=np.float64).reshape(-1)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    v_los_kms = np.asarray(v_los_kms, dtype=np.float64).reshape(-1)
    b_long_G = np.asarray(b_long_G, dtype=np.float64).reshape(-1)

    n = logtau.size
    if not (T.size == n and v_los_kms.size == n and b_long_G.size == n):
        raise ValueError(
            f"logtau, T, v_los_kms, b_long_G must have the same length "
            f"(got {n}, {T.size}, {v_los_kms.size}, {b_long_G.size})"
        )

    v_mic = np.broadcast_to(np.asarray(v_mic_cms, dtype=np.float64), (n,))
    b_x = np.broadcast_to(np.asarray(b_x_G, dtype=np.float64), (n,))
    b_y = np.broadcast_to(np.asarray(b_y_G, dtype=np.float64), (n,))

    if el_p is None:
        # Constant seed; NICOLE recomputes via hydrostatic equilibrium when
        # Impose hydrostatic equilibrium = Y in NICOLE.input.
        el_p_arr = np.full(n, float(el_p_seed), dtype=np.float64)
    else:
        el_p_arr = np.asarray(el_p, dtype=np.float64).reshape(-1)
        if el_p_arr.size != n:
            raise ValueError(f"el_p must have length {n} (got {el_p_arr.size})")

    # Reverse to descending log tau (NICOLE convention)
    order = np.argsort(logtau)[::-1]
    logtau_d = logtau[order]
    T_d = T[order]
    el_p_d = el_p_arr[order]
    v_mic_d = v_mic[order]
    b_long_d = b_long_G[order]
    v_los_d = v_los_kms[order] * 1.0e5  # km/s -> cm/s
    b_x_d = b_x[order]
    b_y_d = b_y[order]

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("Format version: 1.0\n")
        f.write(f"  {v_mac_cms:.3E}    {stray_light:.3f}\n")
        for i in range(n):
            f.write(
                f"  {logtau_d[i]:6.2f}  {T_d[i]:7.1f}  {el_p_d[i]:.3E}  "
                f"{v_mic_d[i]:.3E}  {b_long_d[i]:8.2f}  {v_los_d[i]:.3E}  "
                f"{b_x_d[i]:8.2f}  {b_y_d[i]:8.2f}\n"
            )


def write_binary_model_cube(
    filepath,
    logtau,
    T,
    v_los_kms,
    b_long_G,
    *,
    v_mic_cms=1.0e5,
    v_mac_cms=0.0,
    stray_light=0.0,
    el_p=None,
    el_p_seed=1.0,
    gas_p=None,
):
    # Write a whole (nx, ny) field of model atmospheres in NICOLE's native
    # "nicole2.3bm" binary cube format, so NICOLE can synthesize the entire
    # frame in ONE run_nicole.py invocation (its Fortran loops over pixels
    # internally) instead of one python2 subprocess per pixel. This is the
    # exact inverse of read_model()'s nicole2.3 branch, and run_nicole.py
    # transparently converts nicole2.3 -> its native nicole2.6 (auto-filling
    # default abundances) before handing the cube to the Fortran executable.
    #
    # Record layout (little-endian float64), matching read_model / idl_to_nicole:
    #   header record: b"nicole2.3bm" padded to 16 bytes, int32 nx, int32 ny,
    #                  int64 nz, then zero-padded to (17*nz+3) float64s.
    #   per pixel (iy + ix*ny order): 17 depth variables x nz, then 3 scalars.
    #     var order: z, tau, T, gas_p, rho, el_p, v_los, v_mic, b_los_z,
    #                b_los_x, b_los_y, b_x, b_y, b_z, v_x, v_y, v_z
    #     scalars:   v_mac, stray_frac, expansion
    #   Only tau, T, el_p, v_los(cm/s), v_mic(cm/s), b_los_z(=B_long) are set;
    #   everything else is 0 (matching write_ascii_model's single-pixel output,
    #   NICOLE recomputes gas_p/rho/densities under hydrostatic equilibrium).
    #   If gas_p (nx,ny,nz, dyn/cm^2) is supplied, it is written to the gas_p
    #   slot as well; pair it with Input density=Pgas / Impose hydrostatic
    #   equilibrium=N so NICOLE uses it directly (no HE iteration, hence no
    #   inter-pixel boundary-condition carryover -- each pixel is independent).
    #
    # Inputs T/v_los_kms/b_long_G are (nx, ny, nz); logtau is the shared (nz,)
    # grid (identical for every pixel, as produced by the tau500 remap). v_los
    # is converted km/s -> cm/s. Depth is written descending in logtau (deepest
    # first), matching write_ascii_model.
    import struct

    import numpy as np
    from pathlib import Path

    [int4f, intf, flf] = check_types()

    logtau = np.asarray(logtau, dtype=np.float64).reshape(-1)
    nz = logtau.size
    T = np.asarray(T, dtype=np.float64)
    v_los_kms = np.asarray(v_los_kms, dtype=np.float64)
    b_long_G = np.asarray(b_long_G, dtype=np.float64)
    if not (T.shape == v_los_kms.shape == b_long_G.shape) or T.ndim != 3 or T.shape[2] != nz:
        raise ValueError(
            f"T, v_los_kms, b_long_G must all be (nx, ny, {nz}) with nz matching "
            f"logtau (got {T.shape}, {v_los_kms.shape}, {b_long_G.shape})"
        )
    nx, ny = T.shape[:2]

    if el_p is None:
        el_p_arr = np.full((nx, ny, nz), float(el_p_seed), dtype=np.float64)
    else:
        el_p_arr = np.asarray(el_p, dtype=np.float64)
        if el_p_arr.shape != (nx, ny, nz):
            raise ValueError(f"el_p must be (nx, ny, {nz}) (got {el_p_arr.shape})")

    v_mic_arr = np.broadcast_to(np.asarray(v_mic_cms, dtype=np.float64), (nx, ny, nz))

    if gas_p is None:
        gas_p_arr = None
    else:
        gas_p_arr = np.asarray(gas_p, dtype=np.float64)
        if gas_p_arr.shape != (nx, ny, nz):
            raise ValueError(f"gas_p must be (nx, ny, {nz}) (got {gas_p_arr.shape})")

    # Descending logtau (deepest -> top), applied identically to every pixel.
    order = np.argsort(logtau)[::-1]
    tau_d = logtau[order]

    sizerec = 17 * nz + 3  # float64s per record

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        # Header record (record 0), zero-padded to a full sizerec-float record.
        header = struct.pack("<16s" + int4f + int4f + intf, b"nicole2.3bm", nx, ny, nz)
        f.write(header)
        pad_bytes = sizerec * 8 - len(header)
        f.write(b"\x00" * pad_bytes)

        zero_col = np.zeros(nz, dtype=np.float64)
        rec_fmt = "<" + str(sizerec) + flf
        for ix in range(nx):
            for iy in range(ny):
                cols = [
                    zero_col,                             # z
                    tau_d,                                # tau
                    T[ix, iy, order],                     # T
                    (gas_p_arr[ix, iy, order] if gas_p_arr is not None else zero_col),  # gas_p
                    zero_col,                             # rho
                    el_p_arr[ix, iy, order],              # el_p
                    v_los_kms[ix, iy, order] * 1.0e5,     # v_los (km/s -> cm/s)
                    v_mic_arr[ix, iy, order],             # v_mic
                    b_long_G[ix, iy, order],              # b_los_z (= B_long)
                    zero_col, zero_col,                   # b_los_x, b_los_y
                    zero_col, zero_col, zero_col,         # b_x, b_y, b_z
                    zero_col, zero_col, zero_col,         # v_x, v_y, v_z
                ]
                rec = np.concatenate(cols)
                rec = np.concatenate([rec, [float(v_mac_cms), float(stray_light), 0.0]])
                f.write(struct.pack(rec_fmt, *rec))


def write_prof_cube(filepath, iquv):
    # Write a whole (nx, ny) field of Stokes profiles in NICOLE's native
    # "nicole2.3bp" binary profile-cube format, for use as inversion mode's
    # "Observed profiles=" input (the exact inverse of read_prof's nicole2.3
    # binary branch, and the profile-side counterpart to write_binary_model_cube).
    #
    # Record layout (little-endian float64), matching read_prof/check_prof:
    #   header record: b"nicole2.3bp" padded to 16 bytes, int32 nx, int32 ny,
    #                  int64 nlam, then zero-padded to a full (4*nlam)-float
    #                  record (verified against check_prof's filesize check:
    #                  (4*nlam)*(nx*ny+1)*8 bytes).
    #   per pixel (iy + ix*ny order): 4*nlam floats, interleaved per
    #     wavelength as [I, Q, U, V] (matching read_prof's
    #     .reshape(n_wl, 4) -> stack([:,0],[:,1],[:,2],[:,3]) convention).
    #
    # iquv is (nx, ny, 4, n_wl), Stokes-major/wavelength-minor per pixel
    # (matching NicoleRunner.run_cube's return layout).
    import struct

    import numpy as np
    from pathlib import Path

    [int4f, intf, flf] = check_types()

    iquv = np.asarray(iquv, dtype=np.float64)
    if iquv.ndim != 4 or iquv.shape[2] != 4:
        raise ValueError(f"iquv must be (nx, ny, 4, n_wl) (got {iquv.shape})")
    nx, ny, _, nlam = iquv.shape

    sizerec = 4 * nlam  # float64s per record

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        header = struct.pack("<16s" + int4f + int4f + intf, b"nicole2.3bp", nx, ny, nlam)
        f.write(header)
        pad_bytes = sizerec * 8 - len(header)
        f.write(b"\x00" * pad_bytes)

        rec_fmt = "<" + str(sizerec) + flf
        for ix in range(nx):
            for iy in range(ny):
                # (4, n_wl) -> (n_wl, 4) -> flat [I0,Q0,U0,V0, I1,Q1,U1,V1, ...]
                rec = iquv[ix, iy].T.reshape(-1)
                f.write(struct.pack(rec_fmt, *rec))
