import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import curve_fit
from DOCTData import DOCTData

"""
All code in this file is slightly adapted from Code from R. Morishita et.al., arXiv 2412.09351 (2024)
"""

def computeVLIV(DOCTData, maxTimeWidth = np.nan, compute_VoV = False) -> DOCTData:
    """
    Code from R. Morishita et.al., arXiv 2412.09351 (2024)
    compute LIV curve (VLIV) from time sequential OCT linear intensity.

    Adapted for DOCTData objects:
    - Uses numpy instead of cupy
    - A few redundancies removed for clarity

    Parameters
    ----------
    OctSequence : double (time, x, z) or (time, z, x)
        Time sequence of linear OCT intensity.
        It can be continuous sequence or sparse time sequence.
        
    timePoints : 1D int array (the same size with time of OctSequence)
        indicates the time point at which the frames in the OCT sequence were taken.

    maxTimeWidth : int
        The maximum time width for LIV computation.
        If the LIV curve fitting uses only a particular time-region of LIV curve, 
        it is unnnessesary to compute LIV at the time region exceeding the fitting region.
        With this option, you can restrict the LIV computation time-region and
        can reduce the computation time.
        The default is NaN. If it is default, i.e., maxTimeWidth was not set,
        the full width will be computed.
    compute_VoV: True or False
        Compute variance of all LIVs of identical time window (VoV) : True
        Don't compute VoV : False
        
    Returns
    -------
    VLIV : 3D double (Time points, x, z) or (Time points, z, x)

    possibleMtw : 1D array, int        
        Time points correnspinding to the max-time-window axis of VLIV.
        
    VoV : 3D double (max time window, x, z)
        variance of variances (LIVs) 
    """
    
    # Compute all possible maximum time window
    print(f'Computing VLIV for all possible time windows')
    timePoints = np.arange(DOCTData.data.shape[0])/DOCTData.meta['frame_rate']
    
    # Alternative way to calculate the unique time windows
    possibleMtw = np.abs(np.subtract.outer(timePoints, timePoints))
    # Round to avoid floating point precision issues creating duplicate "unique" values
    possibleMtw = np.round(possibleMtw, decimals=10)
    possibleMtw = np.unique(possibleMtw)
    possibleMtw = possibleMtw[possibleMtw != 0]
    

    # Reduce the time-region to be computed to meet with "maxTimeWidth"
    if np.isnan(maxTimeWidth):
        pass
    else:
        maxTimeWidth = np.asarray(maxTimeWidth)
        possibleMtw = possibleMtw[possibleMtw <= maxTimeWidth] 
    
    logSparseSequence = 10*np.log10(DOCTData.data)
    VLIV = np.zeros((possibleMtw.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2]))

    # variance of variance
    VoV = np.zeros((possibleMtw.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2]))
    i = 0        
    for mtw in tqdm(possibleMtw, desc="Processing each possible mtw:"):
        
        validTimePointMatrix = seekDataForMaxTimeWindow(timePoints, mtw)
        validTimePointMatrix = np.asarray(validTimePointMatrix)    # for cupy computation 
        
        if compute_VoV == True:
            Var = np.zeros((validTimePointMatrix.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2] ))       
        # cupy compute
        for j in range(0, validTimePointMatrix.shape[0]):
            VLIV[i] = VLIV[i] + np.nanvar(logSparseSequence[validTimePointMatrix[j,:]],axis=0)
            # newly added for LIV for each subset 
            if compute_VoV == True:
                Var[j] = np.nanvar(logSparseSequence[validTimePointMatrix[j,:]],axis=0)

        VLIV[i] = VLIV[i] / (validTimePointMatrix.shape[0])
        
        # variance of variance (VoV) at a single time window (2D array)
        if compute_VoV == True:
            VoV[i] =np.var(Var, axis=0)
        
        i = i+1
    
    VLIV = np.asarray(VLIV)   # for numpy computation
    possibleMtw = np.asarray(possibleMtw)
    VoV = np.asarray(VoV)

    return(VLIV, possibleMtw, VoV)

def seekDataForMaxTimeWindow(timePoints, mtw):
    """
    Code from R. Morishita et.al., arXiv 2412.09351 (2024)
    Compute valid combination of timePoints for which the maximum timepoint is smaller than mtw. 
    The valid time sequence will be used to extract
        the OCT frames within the particular time window from whole repeating frames.
        
    Parameters
    ----------
    timePoints : 1D numpy array
        Sequence of time points at which OCT data was taken [in frame time unit]
    mtw : 1D array, int
        The set maximum time window [in frame time unit]

    Returns
    -------
    validTimePointMatrix: 2D numpy array, bool.
        Valid time sequence. VTS[i,:] is the i-th valid time sequence.
    """
    A = np.ones((1,timePoints.shape[0]))
    B = timePoints.reshape(timePoints.shape[0],1)
    timePointMatrix = np.transpose(A*B) - A*B
    validTimePointMatrix = (timePointMatrix <= mtw)*(timePointMatrix >= 0)

    ##---------------------
    ## Let's rewrite later as not to use for-loop
    ##---------------------    
    trueMtw = np.zeros(validTimePointMatrix.shape[0])
    for i in range (0,validTimePointMatrix.shape[0]):
        X = validTimePointMatrix[i,:]
        X = timePoints[X]
        Y = np.max(X) - np.min(X)
        trueMtw[i] = Y
    
    validTimePointMatrix = validTimePointMatrix[(trueMtw >= mtw),:]
    validTimePointMatrix = np.asarray(validTimePointMatrix)
    return(validTimePointMatrix)

def vliv_postprocessing(DOCTData, volumeDataType, frameSeparationTime , 
                frameRepeat, bscanLocationPerBlock: int = 1, blockRepeat: int = 1, blockPerVolume: int = 1, fitting_method = "GPU", 
                       alivInitial = 20, swiftInitial = 1 , bounds = ([0,0],[np.inf, np.inf]), 
                       use_constraint = True , compute_VoV = False, use_weight = False , average_LivCurve = True, motionCorrection = False,
                       octRange = (10., 40.), alivRange =(0., 10.), swiftRange =(0., 3.)):
     
    """
    Code from R. Morishita et.al., arXiv 2412.09351 (2024)
    This function processes aLIV and Swiftness.
    
    Parameters
    ---------
    path_OCT: file path of linear OCT intensity.
        The data type and shape should be "float32" and [time, z, x], or [time, x, z].
    volumeDataType: str
        Either "BSA" or 'Ibrahim2021BOE'
        BSA: Burst scanning protocol
        Ibrahim2021BOE: Dynamic OCT scanning protocol (Same as Ibrahim2021BOE paper)
    frameSeparationTime: constant(float)
        Successive frame measurement time [s] 
    frameRepeat: int
        Number of frames in a single burst
    bscanLocationPerBlock: int
        No of Bscan locations per Block
    blockRepeat:  int 
        Number of block repeats
    blockPerVolume: int 
        Number of blocks in a volume
    fitting_method: str
        Either "CPU" or "GPU"
        CPU: vlivCPUFitExp()
        GPU: vlivGPUFitExp()
    alivInitial: float
        alivInitial = Initial value of a (magnitude) in fitting
    swiftInitial: float
        1/swiftInitial = Initial value of b (time constant) in fitting
    bounds : 2D tuple
        Exploration range of fitting parameters
        ([min a, min b], [max a, max b])
    use_constraint : True or False
        Set bounds of parameters in fitting : True
        don't set bounds : False
    compute_VoV : True or False
        Compute variance of all LIVs of identical time window (VoV) for test (debug) and weighted fitting: True
        Don't compute VoV : False
    use_weight : True or False
        Apply weight for fitting : True
        Make sure if "compute_VoV = True". VoV will be used as the weight.
        The weighted fitting has been implemented only in GPU fitting (fitting_method = "GPU").
        Don't apply weight : False
    average_LivCurve : True or False
        Average LIV curve before fitting using 3*3 kernel for increasing accuracy: True
        Don't average LIV curve : False
    motionCorrection: True or False
        Apply motion correction (DOI: 10.1364/BOE.488097): True
        Don't apply: False
    octRange: 1D tuple (min, max)
        dynamic range of dB-scaled OCT intensity, which is used as brightness of pseudo-color image
    alivRange: 1D tuple (min, max)
        dynamic range of aLIV, which is used as hue of pseudo-color image
    swiftRange:1D tuple (min, max)
        dynamic range of Swiftness, which is used as hue of pseudo-color image

    Output
    -----------
    dB-scaled OCT intensity: 3D gray scale image
    VLIV: 4D double [timePoints, slowscanlocations, z, x]
        LIV array at different time windows (LIV curve)
    timewindows: 2D double
    aLIV: 3D gray scale and RGB image
    Swiftness: 3D gray scale and RGB image
    
    """
    print(f"Now computing aLIV and swiftness from R. Morishita et.al., arXiv 2412.09351 (2024)")
    maxFrameNumber = frameRepeat * blockRepeat * bscanLocationPerBlock # Maximum frame number per Block 
    burstingPeriod = frameRepeat *  bscanLocationPerBlock # Peroid of burst sampling
    numLocation = bscanLocationPerBlock * blockPerVolume # Number of slow scan locations
    
    print(f"Processing: {DOCTData.data}")
    
    ## OCT intensity
    octFrames = DOCTData.copy(deep=True)
    height = octFrames.data.shape[1]
    width = octFrames.data.shape[2]

    aliv = np.zeros((numLocation, height, width), dtype=np.float32)
    swift = np.zeros((numLocation, height, width),  dtype=np.float32)
    oct_db = np.zeros((numLocation, height, width),  dtype=np.float32)
    # Use mean intensity across time for the brightness reference
    oct_db[0] = np.mean(octFrames.data, axis=0)

    for floc in range(0,numLocation):

        sparseSequence = octFrames
        timePoints = np.arange(octFrames.data.shape[0])/octFrames.meta['frame_rate']
        
        if floc == 0: #for save VLIV array
            VLIV_save = np.zeros((numLocation, timePoints.shape[0]-1, height, width),  dtype=np.float32)
            if compute_VoV == True:
                VoV_save = np.zeros((numLocation, timePoints.shape[0]-1, height, width),  dtype=np.float32)
        ## Compute VLIV
        VLIV , possibleMtw , VoV = computeVLIV(sparseSequence, maxTimeWidth =  np.nan, compute_VoV = False)
        
        ## Average LIV curve
        if average_LivCurve == True:
            twIdx = 0
            for twIdx in range(0, VLIV.shape[0]):                
                VLIV[twIdx,:,:] = cv2.blur(VLIV[twIdx,:,:], (3,3))
                twIdx = twIdx + 1

        ## curve fitting on LIV curve to compute saturation level (magnitude) and time constant (tau)
        if fitting_method == 'GPU':
            mag, tau = vlivGpuFitExp(VLIV, possibleMtw, VoV, frameSeparationTime, alivInitial, swiftInitial, bounds, use_constraint, use_weight)
        
        if fitting_method == 'CPU':
            print(f"  - VLIV shape: {VLIV.shape}, range: [{np.nanmin(VLIV):.6f}, {np.nanmax(VLIV):.6f}]")
            print(f"  - possibleMtw shape: {possibleMtw.shape}, range: [{possibleMtw[0]:.6f}, {possibleMtw[-1]:.6f}]")
            mag, tau = vlivCPUFitExp(VLIV, possibleMtw, frameSeparationTime, alivInitial, swiftInitial, bounds, use_constraint = False)
            print(f"  - mag range: [{np.nanmin(mag):.6f}, {np.nanmax(mag):.6f}]")
            print(f"  - tau range: [{np.nanmin(tau):.6f}, {np.nanmax(tau):.6f}]")

        aliv [floc] = mag ## aLIV
        swift [floc] = 1/ tau ## Swiftness
        VLIV_save[floc,:,:,:] = VLIV ## LIV curve (VLIV)
        if compute_VoV == True:
            VoV_save[floc,:,:,:] = VoV
            
    ## Generate each path save data
    #suffix_intensity_linear = ".tiff"
    #root = path_OCT[:-len(suffix_intensity_linear)]
    #path_vliv = root  +  '_vliv.npy'
    #path_timewindow = root  +  '_timewindows.npy'
    #path_vov = root + '_vov.npy'
    ### Save the gray scale image of dB-scaled OCT intensity
    #tifffile.imwrite(root  +  '_dbOct.tif', oct_db )
    ### Save time windows, LIV curve (VLIV), and variance of all LIVs for each time window (VoV)
    #np.save(path_timewindow, possibleMtw)
    #np.save(path_vliv, VLIV_save)
    #if compute_VoV == True:
    #    np.save(path_vov, VoV_save)

    
    ## Convert to color aLIV and Swiftness images
    #aliv_rgb = generate_RgbImage(doct = aliv, dbInt = oct_db, doctRange = alivRange, octRange = octRange)
    #swift_rgb = generate_RgbImage(doct = swift, dbInt = oct_db, doctRange = swiftRange, octRange = octRange)
    
    # Store both raw values and RGB images
    DOCTData.results['aliv_raw'] = aliv
    DOCTData.results['swift_raw'] = swift
    #DOCTData.results['aliv'] = aliv_rgb
    #DOCTData.results['swiftness'] = swift_rgb

    ## Save the gray scale and rgb images of aLIV and Swiftness
    #path_aliv = root  +  '_aliv.tif'
    #path_aliv_view = root + f'_aliv_min{alivRange[0]}-max{alivRange[1]}.tif'

    #path_swift = root  + '_swiftness.tif'
    #path_swift_view = root + f'_swiftness_min{swiftRange[0]}-max{swiftRange[1]}.tif'

    #tifffile.imwrite(path_aliv, aliv)  
    #tifffile.imwrite(path_aliv_view, (aliv_rgb*255).astype(np.uint8)) 

    #tifffile.imwrite(path_swift, swift)
    #tifffile.imwrite(path_swift_view, (swift_rgb*255).astype(np.uint8))
    
    return DOCTData
    print("VLIV Processing Ended")


def vlivGpuFitExp(VLIV, possibleMtw, VoV, frameSeparationTime, mfInitial, dfInitial,
                  bounds = ([0,0],[np.inf, np.inf]), use_constraint = False, use_weight = False):
    """
    Code from R. Morishita et.al., arXiv 2412.09351 (2024)
    Provide saturation level (magnitude) and time constant (tau)
    from LIV curve (VLIV) by exponential GPU-based fitting.

    Parameters
    ----------
    VLIV : 3D double array, (time window, z, x) 
        LIV curve (VLIV)
    possibleMtw : 1D int array
        time window indicators for VLIV data array.
    VoV : 3D double array, (time window, z, x)
        variance of varianves (LIVs)
    frameSeparationTime: constant (float)
        Successive frame measurement time [s] 
    alivInitial: float
        alivInitial = Initial value of a in fitting
    swiftInitial: float
        1/swiftInitial = Initial value of b in fitting
    bounds : 2D tuple
        ([min a, min b], [max a, max b])
    use_constraint : True or False
        Set bounds of parameters in fitting : True
        Don't set bounds : False
    use_weight : True or False
        Apply weight for fitting : True
        Don't apply weight : False
        

    Returns
    -------
    mag : 2d double (z, x)
        magnitude of the 1st order saturation function.
    tau : 2d doubel (z, x)
        time constant of the 1st-order saturation function.
    
    """
    ## Roll the axis of VLIV such that the number of points should be at the last axis in order to fit using GPUfit
    VLIV = np.rollaxis(VLIV, 0, 3)
    height = VLIV.shape[0]
    width = VLIV.shape[1]
    n_points = VLIV.shape[2]
        
    ## Reshape to 1D array 
    VLIV_re = np.reshape(VLIV,  (-1, n_points))
    ## Amount of LIV curves to be fitted and fitting parameters
    n_fits = height * width
    n_parameter = 2  
    
    ## tolerance (smaller tolerance -> better fitting accuracy)
    tolerance = 1.0E-6
        
    # max_n_iterations
    max_n_iterations = 100
        
    # model id
    model_id = gf.ModelID.SATURATION_1D
        
    ## initial parameters
    initial_parameters = np.ones((n_fits, n_parameter), dtype=np.float32)
    initial_parameters[:, 0] = mfInitial 
    initial_parameters[:, 1] = 1/dfInitial
    
    ## set exploration range of fitting parameters, which will be used in gpufit_constrained()
    if use_constraint == False : pass
    else :
        constraints = np.ones((n_fits, 2*n_parameter), dtype=np.float32)
        constraints[:,0] = bounds[0][0]
        constraints[:,1] = bounds[1][0]
        constraints[:,2] = bounds[0][1]
        constraints[:,3] = bounds[1][1]
        constraint_types = np.ones((n_parameter), dtype=int)
        constraint_types[:] = 3
    
    ## calculate weight for fitting based on VoV (variance of all LIVs averaged)
    if use_weight == False : pass
    else :
        ##--------------------------------------------
        ## weight candudate-1 (variance of all LIVs)
        ##--------------------------------------------
        weight = 1/VoV # weight for fitting
        ##------------------------------------------------------------------------
        ## weight candidate-2 (data dependency corrected variance of unbiased LIV)
        ##------------------------------------------------------------------------
        # NoS2V = possibleMtw/possibleMtw[0] + 1 # NoS2V : number of samples to compute Variance
        # VoUL = VoV * ((np.multiply(NoS2V, 1/(NoS2V-1)))**2)[:, np.newaxis, np.newaxis] # varianve of unbiased LIV
        # CVoUL = VoUL * (NoS2V/32)[:,np.newaxis, np.newaxis] # data dependency corrected variance of unbiased LIV (CVoUL)
        # weight = (1/CVoUL) # weight for fitting
    
    timeWindow = np.zeros(len(possibleMtw))
    for i in range (len(possibleMtw)):
        timeWindow[i] = possibleMtw[i] * frameSeparationTime
        
    if use_constraint and use_weight : #use_constraint == True & use_weight == True
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit_constrained(np.ascontiguousarray(VLIV_re, dtype='float32'), weight , model_id, \
                                                                                          initial_parameters, constraints, constraint_types, tolerance, \
                                                                                          max_n_iterations, None, None, timeWindow.astype(np.float32))
    if use_constraint and not use_weight : #use_constraint == True & use_weight == False
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit_constrained(np.ascontiguousarray(VLIV_re, dtype='float32'), None, model_id, \
                                                                                      initial_parameters, constraints, constraint_types, tolerance, \
                                                                                      max_n_iterations, None, None, timeWindow.astype(np.float32))         
    if not use_constraint and use_weight : # use_constraint == False & use_weight == True 
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit(np.ascontiguousarray(VLIV_re, dtype='float32'), weight, model_id, \
                                                                                      initial_parameters, tolerance, \
                                                                                      max_n_iterations, None, None, timeWindow.astype(np.float32))
    if not use_constraint and not use_weight : #(default) use_constraint == False & use_weight == False
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit(np.ascontiguousarray(VLIV_re, dtype='float32'), None, model_id, \
                                                                                      initial_parameters, tolerance, \
                                                                                      max_n_iterations, None, None, timeWindow.astype(np.float32))
    ## reshape 1D array to 2D array
    mag = parameters [:, 0].reshape (height, width)
    tau = parameters [:, 1].reshape (height, width)
    
    return (mag, tau)


def vlivCPUFitExp(VLIV, possibleMtw, frameSeparationTime, alivInitial, swiftInitial,
                  bounds = ([0,0],[np.inf, np.inf]), use_constraint = False):
    """
    Code from R. Morishita et.al., arXiv 2412.09351 (2024)
    Provide saturation level (magnitude) and time constant (tau)
    from LIV curve (VLIV) by exponential CPU-based fitting.

    Parameters
    ----------
    VLIV : 3D double array, (time window, z, x)
        LIV curve (VLIV)
    possibleMtw : 1D int array
        time window indicators for VLIV data array.
    frameSeparationTime: constant (float)
        Successive frame measurement time [s] 
    alivInitial: float
        alivInitial = Initial value of a in fitting
    swiftInitial: float
        1/swiftInitial = Initial value of b in fitting
    bounds : 2D tuple
        bounds for fitting
        ([min a, min b], [max a, max b])
    use_constraint : True or False
        Set bounds of parameters in fitting : True
        don't set bounds : False

    Returns
    -------
    mag : 2d double (z, x)
        magnitude of the 1st order saturation function.
    tau : 2d doubel (z, x)
        time constant of the 1st-order saturation function.
    
    """
    height = VLIV.shape[1]
    width = VLIV.shape[2]
    mag = np.empty((height, width))
    tau = np.empty((height, width))

    def saturationFunction(x, a, b):
        return(np.absolute(a)*(1-(np.exp(-x/b))))
    
    for depth in tqdm(range(0, int(height)), desc="Fitting curves (rows)"):
      
        for lateral in range(0, int(width)):
            LivLine = VLIV[:,depth, lateral]
            ## Remove nan from LivLine (and also from corresponding possibleMtw).
            nonNanPos = (np.isnan(LivLine) == False)
            if np.sum(nonNanPos) >= 2:
                LivLine2 = LivLine[nonNanPos]
                t = possibleMtw[nonNanPos]
                ## 1D list of time window in frame units -> 1D list of time window in second unit.
                for i in range (len(t)):
                    t[i] = t[i] * frameSeparationTime

                try:
                    if use_constraint == False:
                        popt, pcov = curve_fit(saturationFunction, 
                                           t,
                                           LivLine2,
                                            method = "lm", # when no boundary, "lm"
                                           p0 = [alivInitial, 1/swiftInitial] )
                    else: 
                        popt, pcov = curve_fit(saturationFunction, 
                                           t,
                                           LivLine2,
                                            method = "dogbox",# when add boundary, "dogbox"
                                           p0 = [alivInitial, 1/swiftInitial],
                                           bounds = bounds)# set boundary [min a, min b],[max a, max b]
                except RuntimeError:
                    mag[depth, lateral] = np.nan
                    tau[depth, lateral] = np.nan
                    
                mag[depth, lateral] = popt[0]
                tau[depth, lateral] = popt[1]

            else:
                mag[depth, lateral] = np.nan
                tau[depth, lateral] = np.nan
    
    return(mag, tau)