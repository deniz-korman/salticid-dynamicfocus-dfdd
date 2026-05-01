from scipy import signal, interpolate, fft
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
import pdb
import random
import time
import sys
import glob
import cv2
import pickle
import os
import math
from pathlib import Path
import csv


np.set_printoptions(threshold=sys.maxsize)

# GLOBAL VARIABLES DESCRIBING THE VISUAL SYSTEM
# Obtain from the measurement in the real world
SIGMA = 0.00039 # in meters Habronattus pyrritrix average lens diameter 390 micron 

# Used for initialising Camera class
DEPTH = np.inf

# Resolution of the visual system, used for PSF and Laplacian calculations
PIXEL_PITCH = 8.423e-7 # in m based on 0.07 deg rhabdomere angular separation for Habronattus pyrritrix
                       # Derived from (0.07 * (pi/180) * f)), where f=689.42

# Key image processing parameters based on previous implementations of DFDD in camera systems. 
SENSOR_RADIUS = 300
TEXTURE_RADIUS = SENSOR_RADIUS
SENSOR_PANEL_XX, SENSOR_PANEL_YY = np.meshgrid(
    np.linspace(-SENSOR_RADIUS, SENSOR_RADIUS, 2 * SENSOR_RADIUS + 1) * PIXEL_PITCH,
    np.linspace(-SENSOR_RADIUS, SENSOR_RADIUS, 2 * SENSOR_RADIUS + 1) * PIXEL_PITCH,
)
ERROR_NUMBER = 0.0000000001
SIGMA0 = 0.0001
LAPLACIAN = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]) / (
    PIXEL_PITCH**2
)
GAUSSIAN = np.array(
    [
        [0.0, 0.0013, 0.004, 0.0013, 0.0],
        [0.0013, 0.0377, 0.1162, 0.0377, 0.0013],
        [0.004, 0.1162, 0.3579, 0.1162, 0.004],
        [0.0013, 0.0377, 0.1162, 0.0377, 0.0013],
        [0.0, 0.0013, 0.004, 0.0013, 0.0],
    ]
)

# Plot parameters
font = {
    "size": 32,
}
matplotlib.rc("font", **font)


def fspecial(XX, YY, param, shape="disk"):
    step = XX[0, 1] - XX[0, 0]
    if shape == "gaussian":
        # Gaussian PSF do not require heavy upsampling
        oversampling_rate = 1
        oversampling_step = step / oversampling_rate
        sigma = param["sigma"]
        radius = sigma * 5
        tab = np.ceil(radius / oversampling_step)
        xx_, yy_ = np.meshgrid(
            np.arange(-tab, tab + 1) * oversampling_step,
            np.arange(-tab, tab + 1) * oversampling_step,
        )

        oversampling_psf = np.exp(-(xx_**2 + yy_**2) / (2 * sigma**2))
        oversampling_psf = oversampling_psf / (2 * np.pi * (sigma / step) ** 2)

    elif shape == "smooth-disk":
        # Gaussian PSF do not require heavy upsampling
        if "order" in param:
            order = param["order"]
        else:
            order = 5.5
        # oversampling causes inaccurate depth prediction and we don't know why
        oversampling_rate = 1
        oversampling_step = step / oversampling_rate
        sigma = param["sigma"]
        radius = sigma * 3
        tab = np.ceil(radius / oversampling_step)
        xx_, yy_ = np.meshgrid(
            np.arange(-tab, tab + 1) * oversampling_step,
            np.arange(-tab, tab + 1) * oversampling_step,
        )

        oversampling_psf = np.exp(
            -np.power((np.abs(xx_) ** 2 + np.abs(yy_) ** 2) / (2 * sigma**2), order / 2)
        )
        oversampling_psf = oversampling_psf / np.sum(oversampling_psf)

    elif shape == "disk":
        # 1. create an oversampled disk
        oversampling_rate = 1
        oversampling_step = step / oversampling_rate
        radius = param["radius"]
        tab = np.ceil(radius / oversampling_step)
        xx_, yy_ = np.meshgrid(
            np.arange(-tab, tab + 1) * oversampling_step,
            np.arange(-tab, tab + 1) * oversampling_step,
        )

        oversampling_psf = ((xx_**2 + yy_**2) <= radius**2).astype(np.float32)
        oversampling_psf = oversampling_psf / (np.pi * (radius / step) ** 2)

    elif shape == "pillbox":
        # Gaussian PSF do not require heavy upsampling
        order = 4
        # oversampling causes inaccurate depth prediction and we don't know why
        oversampling_rate = 1
        oversampling_step = step / oversampling_rate
        sigma = param["sigma"]
        radius = sigma * 3
        tab = np.ceil(radius / oversampling_step)
        xx_, yy_ = np.meshgrid(
            np.arange(-tab, tab + 1) * oversampling_step,
            np.arange(-tab, tab + 1) * oversampling_step,
        )

        oversampling_psf = np.exp(
            -np.power((np.abs(xx_) ** 2 + np.abs(yy_) ** 2) / (2 * sigma**2), order / 2)
        )
        oversampling_psf = oversampling_psf / np.sum(oversampling_psf)

    elif shape == "multiple-pillbox":
        # 1. create an oversampled disk
        order = 20
        # oversampling causes inaccurate depth prediction and we don't know why
        oversampling_rate = 1
        oversampling_step = step / oversampling_rate
        sigma = param["sigma"]
        radius = sigma * 3
        tab = np.ceil(radius / oversampling_step)
        xx_, yy_ = np.meshgrid(
            np.arange(-tab, tab + 1) * oversampling_step,
            np.arange(-tab, tab + 1) * oversampling_step,
        )

        num_pillbox = 3

        new_sigma = sigma / num_pillbox
        oversampling_psf = np.zeros(xx_.shape)
        for i in range(num_pillbox):
            # current_oversampling_psf = (
            #     ((xx_ - radius + newRadius + 2 * i * newRadius) ** 2 + yy_**2)
            #     <= newRadius**2
            # ).astype(np.float32)
            current_oversampling_psf = np.exp(
                -np.power(
                    (
                        np.abs(xx_ - sigma + new_sigma + 2 * i * new_sigma) ** 2
                        + np.abs(yy_) ** 2
                    )
                    / (2 * (new_sigma * 0.67) ** 2),
                    order / 2,
                )
            ).astype(np.float32)
            # plt.imshow(current_oversampling_psf)
            # plt.show()
            oversampling_psf = oversampling_psf + current_oversampling_psf

        oversampling_psf = oversampling_psf / oversampling_psf.sum()

    # 2. convolve with a box filter to simulate the response field of a pixel
    box = (
        np.ones((oversampling_rate, oversampling_rate), np.float32)
        / oversampling_rate**2
    )
    # tic = time.perf_counter()
    filtered_disk = signal.fftconvolve(oversampling_psf, box, mode="same")
    # toc = time.perf_counter()
    # print(f"Get normalised PSF in {toc - tic:0.4f} seconds")

    # 3. interpolate
    f = interpolate.RegularGridInterpolator(
        (yy_[:, 0], xx_[0, :]),
        filtered_disk,
        method="nearest",
        bounds_error=False,
        fill_value=0,
    )
    final_disk = f((YY, XX))

    if shape == "multiple-pillbox":
        return final_disk / np.sum(final_disk) / 3
    return final_disk / np.sum(final_disk)


class Texture:
    def __init__(self) -> None:
        return

    def setBrightness(self, brightness):
        self.brightnessArray = brightness

    def setDepth(self, depth):
        self.depth = depth

    def setsigma(self, sigma):
        self.sigma = sigma


class Camera:
    def __init__(
        self,
        PSF_shape: str,
        optical_power,
        retinalDistance,
        Sigma,
        order=5.5,
    ) -> None:
        self.PSF_shape = PSF_shape
        self.optical_power = optical_power
        self.retinalDistance = retinalDistance
        self.Sigma = Sigma
        self.order = order

    def setTexture(self, texture: Texture):
        self.texture = texture

    def getOffsetAndsigma(self):
        self.texture.setsigma(
            1
            - self.retinalDistance * self.optical_power
            + self.retinalDistance / self.texture.depth
        )

    def getPSF(self,save_img=False,savePSFlabel=''):
        sigma = self.texture.sigma
        brightness = self.texture.brightnessArray
        sizeDifference = SENSOR_RADIUS - TEXTURE_RADIUS
        sharpTexture = np.pad(
            brightness,
            ((sizeDifference, sizeDifference), (sizeDifference, sizeDifference)),
        )

        # generate PSF
        sigma = np.sqrt((sigma**2) * self.Sigma**2)
        if (
            self.PSF_shape == "gaussian"
            or self.PSF_shape == "smooth-disk"
            or self.PSF_shape == "multiple-pillbox"
            or self.PSF_shape == "pillbox"
        ):
            param = {"sigma": sigma, "order": self.order}
        elif self.PSF_shape == "disk":
            param = {"radius": sigma}
        self.normalisedPSF = fspecial(
            SENSOR_PANEL_XX, SENSOR_PANEL_YY, param, self.PSF_shape
        )

        # convolve with texture
        texturePSF = signal.fftconvolve(self.normalisedPSF, sharpTexture, "same")

        if save_img:
            file_name = f"./convolved_imgs/{savePSFlabel}sigma{sigma:.1e}_depth{self.texture.depth:.2f}.png"
            file_path = Path(file_name)
            if not file_path.exists():
                plt.imsave(
                    file_name,
                    texturePSF,
                    vmin=0,
                    vmax=255,
                    cmap="gray",
                )
        # plt.imsave(
        #     f"./PSF{self.texture.depth:.2f}.png", self.normalisedPSF, cmap="gray"
        # )

        self.PSF = texturePSF

    def calculatePSF(self,savePSFimage=False,img_label=''):
        self.getOffsetAndsigma()
        self.getPSF(save_img=savePSFimage,savePSFlabel=img_label)


class CameraDifferential:
    def __init__(
        self,
        cameraSystem: Camera,
        delta_rho=0,
        delta_Sigma=0,
        ratio=1.0,
    ) -> None:
        if not hasattr(cameraSystem, "PSF"):
            cameraSystem.calculatePSF()
        self.cameraSystem = cameraSystem
        self.delta_rho = delta_rho
        self.delta_Sigma = delta_Sigma
        self.ratio = ratio

        # blur filter
        if SIGMA0 != 0:
            param = {"sigma": SIGMA0}
            HALF_TAB = SIGMA0 / PIXEL_PITCH * 5
            XX, YY = np.meshgrid(
                np.arange(-HALF_TAB, HALF_TAB + 1) * PIXEL_PITCH,
                np.arange(-HALF_TAB, HALF_TAB + 1) * PIXEL_PITCH,
            )
            self.blurKernel = fspecial(XX, YY, param, "gaussian")
        else:
            self.blurKernel = None


def uniformFiltering(img, ksize):
    kernel = np.ones((ksize, ksize))
    kernel = kernel / np.sum(kernel)
    result = signal.fftconvolve(img, kernel, "same")

    return result


def getIxIy(img, ksize=1, ktype="None"):
    xKernel = np.array([[-0.5, 0, 0.5]])
    yKernel = np.array([[-0.5], [0], [0.5]])

    if ktype == "Gaussian":
        Ix = cv2.GaussianBlur(img, (ksize, ksize), 0, borderType=cv2.BORDER_REFLECT)
        Iy = cv2.GaussianBlur(img, (ksize, ksize), 0, borderType=cv2.BORDER_REFLECT)
    elif ktype == "Uniform":
        Ix = uniformFiltering(img, ksize)
        Iy = uniformFiltering(img, ksize)
    else:
        Ix = img
        Iy = img

    Ix = signal.convolve2d(Ix, xKernel, "same", "symm")
    Iy = signal.convolve2d(Iy, yKernel, "same", "symm")

    return Ix, Iy


def getConfidenceMapByIrho(I_rho_t):
    confidenceMap = I_rho_t**2
    return confidenceMap


def filterResultByConfidence(
    ZArray, ZConfidence, working_range,confidence_level=0.95, ):
    if confidence_level == 0:
        return np.where(
            (ZArray < max(working_range)) & (ZArray > min(working_range)),
            ZArray,
            np.nan,
        )
        return np.array(ZArray)

    ZConfidence_ = ZConfidence.flatten()
    ZConfidence_ = ZConfidence_[ZConfidence_ < np.inf]
    ZConfidence_f = np.where(ZConfidence < np.inf, ZConfidence, np.nan)
    sortZkfConfidence = np.sort(ZConfidence_)

    if sortZkfConfidence.size == 0:
        ZArray[:] = np.nan
        return ZArray

    confidenceLevel = sortZkfConfidence[
        int((len(sortZkfConfidence) - 1) * confidence_level)
    ]

    ZArray_ = np.where(
        (ZArray < max(working_range)) & (ZArray > min(working_range)), ZArray, np.nan
    )

    return np.where(ZConfidence_f > confidenceLevel, ZArray_, np.nan)


def getDepthMap_FT(I, I_rho_t, params,
                           scales=3, Laplacian_filter=LAPLACIAN, LaplacianI=None,):
    '''
    Implements DFD Based on the Focal Track Algorithm
    Guo, Q., Alexander, E., Zickler, T.: Focal Track: Depth and Accommodation
    with Oscillating Lens Deformation. In: International Conference on Computer
    Vision (ICCV). IEEE (2017).
    Adapted from: https://github.itap.purdue.edu/guo-research-group/FocalTrack_forTesting
    '''
    rho = params["rho"]
    Sigma = params["Sigma"]
    Delta_rho = params["Delta_rho"]
    retinalDistance = params["retinalDistance"]

    if LaplacianI is None:
        LaplacianI = signal.fftconvolve(I, Laplacian_filter, "same")

    VarGamma = Sigma * Sigma * retinalDistance * retinalDistance * Delta_rho
    VarEpsilon = Sigma * Sigma * retinalDistance * Delta_rho * (retinalDistance * rho - 1)
    VarZeta = 1

    current_I_rho_t = I_rho_t
    current_LaplacianI = LaplacianI
    height, width = I.shape
    list_ZMap = []
    list_ConfidenceMap = []

    for i in range(scales):
        current_V = VarGamma * current_LaplacianI
        current_W = VarEpsilon * current_LaplacianI + VarZeta * current_I_rho_t

        current_V = cv2.resize(
            current_V, (width, height), interpolation=cv2.INTER_LINEAR
        )
        current_W = cv2.resize(
            current_W, (width, height), interpolation=cv2.INTER_LINEAR
        )
        Vx, Vy = getIxIy(current_V)
        Wx, Wy = getIxIy(current_W)
        list_VW = [[current_V, current_W], [Vx, Wx], [Vy, Wy]]

        for VW in list_VW:
            V = VW[0]
            W = VW[1]

            ZMap = np.divide(V, W, out=np.zeros_like(V), where=W != 0)

            ConfidenceMap = I_rho_t**2

            list_ZMap.append(ZMap)
            list_ConfidenceMap.append(ConfidenceMap)

        # process the data for next scale
        current_I_rho_t = signal.fftconvolve(current_I_rho_t, GAUSSIAN, "same")
        current_LaplacianI = signal.fftconvolve(current_LaplacianI, GAUSSIAN, "same")
        current_I_rho_t = cv2.resize(
            current_I_rho_t,
            (current_I_rho_t.shape[1] // 2, current_I_rho_t.shape[0] // 2),
            interpolation=cv2.INTER_LINEAR,
        )
        current_LaplacianI = cv2.resize(
            current_LaplacianI,
            (current_LaplacianI.shape[1] // 2, current_LaplacianI.shape[0] // 2),
            interpolation=cv2.INTER_LINEAR,
        )

    ZMap_all = np.array(list_ZMap)
    ConfidenceMap_all = np.array(list_ConfidenceMap)
    w_softmax = np.exp(ConfidenceMap_all) / np.sum(
        np.exp(ConfidenceMap_all), axis=0, keepdims=True
    )
    result_ZMap = np.sum(ZMap_all * w_softmax, axis=0)
    result_ConfidenceMap = np.sum(ConfidenceMap_all * w_softmax, axis=0)

    return result_ZMap, result_ConfidenceMap, I_rho_t / LaplacianI


def getDepthMap_FS(I, I_rho_t, params, kernelSize=5,
                   Laplacian_filter=LAPLACIAN, LaplacianI=None,):
    '''
    Implements DFD Based on the Focal Split Processing Algorithm
    https://arxiv.org/abs/2504.11202
    https://github.itap.purdue.edu/guo-research-group/FocalSplit_clean
    '''
    rho = params["rho"]
    Sigma = params["Sigma"]
    delta_rho = params["Delta_rho"]
    retinalDistance = params["retinalDistance"]

    if LaplacianI is None:
        LaplacianI = signal.fftconvolve(I, Laplacian_filter, "same")

    varAlpha = - Sigma**2
    varBeta = - Sigma**2 * (rho - 1/retinalDistance)

    V = varAlpha * LaplacianI
    W = varBeta * LaplacianI + I_rho_t

    if kernelSize > 1:
        kernel = np.ones((kernelSize, 1))
        VW = V * W
        square_W = W**2
        VW = signal.convolve2d(VW, kernel, "same", "symm")
        VW = signal.convolve2d(VW, kernel.T, "same", "symm")
        square_W = signal.convolve2d(square_W, kernel, "same", "symm")
        square_W = signal.convolve2d(square_W, kernel.T, "same", "symm")
        ZMap = np.divide(VW, square_W, out=np.zeros_like(VW), where=square_W != 0)
    else:
        ZMap = np.divide(V, W, out=np.zeros_like(V), where=W != 0)

    return ZMap


def getDfDDErrors_multiMethod(params, DFD_model='FocalSplit', fieldCurvature = True,
                                    surface_count=3,confidenceLevel=0,overwrite=False,
                                    plot_indiv=True, plot_dist=True, save_plot=False,
                                    export_MAEs=False):
    rho = params["rho"]
    Delta_rho = params["Delta_rho"]
    Sigma = params["Sigma"]
    retinalDistance = params["retinalDistance"]
    retinalGapsize = params["retinalGapsize"]
    focusingDistance = params["focusingDistance"]
    initial_focusingDistance_str = params["focusingDistance_tag"]
    model_str = f'{DFD_model}-{str(surface_count-1)}dfComparisons-'
    PSF_shape = params["PSF_shape"]

    kernelSize = params["kernelSize"]


    camera = Camera(PSF_shape, rho, retinalDistance, Sigma)

    paths = [
        "../img/habitat1.png",
        "../img/habitat2.png",
        "../img/habitat3.png",
        "../img/habitat4.png",
        "../img/habitat5.png",
        "../img/texture1.png",
        "../img/texture2.png",
        "../img/texture3.png",
        "../img/texture4.png",
        "../img/texture5.png",
        "../img/texture6.png",
        "../img/texture7.png",
        "../img/texture8.png",
        "../img/texture9.png",
        "../img/texture10.png",
        "../img/texture11.png",
        "../img/texture12.png"
    ]

    errors = []
    errors_95 = []


    #Import eye-angles used
    with open(f'normalized_field_curvature.csv', mode='r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        theta_deg = [round(float(rows[0]),5) for rows in reader]

    theta = [x*(math.pi/180) for x in theta_deg]
    ground_distance_m = [(params["h"]/np.tan(x)) for x in theta]

    # Import field curvature data and update model_name for output
    if fieldCurvature:
        # If importing a different set of field curvature data, ensure that the
        # You have a matching value for every angle value you are evaluating in theta_deg
        with open(f'/normalized_field_curvature.csv', mode='r') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            # Contains angle-based focal length shrinkage ratio when deviating from on-axis viewing
            loess_dict = {round(float(rows[0]),5):float(rows[1]) for rows in reader}
            model_str = model_str + 'wFieldCurvature'


    pkl_output = f"{model_str}_HAPYParamsWLensPitDistance-avgTierIdists_{initial_focusingDistance_str}Focus_MAE.pkl"
    if (not os.path.exists(pkl_output)) or (overwrite):
        for path in paths:
            print("Current texture:%s" % path)

            texture = Texture()
            img_name = Path(path).stem
            image = Image.open(path)
            image = image.resize([2 * TEXTURE_RADIUS + 1, 2 * TEXTURE_RADIUS + 1])
            imageArray = np.array(image)
            texture.setBrightness(imageArray)
            texture.setDepth(DEPTH)
            camera.setTexture(texture)
            camera.calculatePSF()


            error = []
            error_95 = []

            # Vary viewing angle from looking at horizon to looking 30 deg down
            for i in range(len(ground_distance_m)):
                depth = ground_distance_m[i] # Where gaze intersects with the ground
                angle = theta_deg[i]         # Viewing angle.
                print("Current depth:%f" % depth)

                # Update optics of the system at each angle if accounting for field curvature
                if fieldCurvature:
                    # The effective focal length of lens under current field curvature position.
                    fieldCurvatureDistance = retinalDistance * loess_dict[round(angle,5)]

                    # Calculate & update optical power for Tier i of retina & optical power difference between two tiers.
                    rho = 1/fieldCurvatureDistance + 1/focusingDistance
                    Delta_rho = (1/(fieldCurvatureDistance-retinalGapsize) + 1/focusingDistance) - (1/fieldCurvatureDistance + 1/focusingDistance)
                    params["rho"] = rho
                    params["Delta_rho"] = Delta_rho

                # rho at secondary surface(s)
                rhoPlus = rho + Delta_rho
                rhoMinus = rho - Delta_rho

                # Set the depth of target object
                camera.texture.setDepth(depth)

                #calculate PSF when viewing object with retinal tier i
                camera.optical_power = rho
                camera.Sigma = Sigma
                camera.calculatePSF(savePSFimage=True,img_label=f'{model_str}_init{initial_focusingDistance_str}Focus_{str(angle)}deg_{img_name}_')
                img = camera.PSF

                #calculate PSF when viewing object with retinal tier ii, 
                camera.optical_power = rhoMinus
                camera.Sigma = Sigma
                camera.calculatePSF()
                img_rhoMinus = camera.PSF

                # calculate PSF when viewing object with hypothethical surface positioned behind tier i
                # mimicking a mirror reflection of tier ii. This is used to operate DFD methods that evalutate defocus in both directions.
                camera.optical_power = rhoPlus
                camera.Sigma = Sigma
                camera.calculatePSF()
                img_rhoPlus = camera.PSF

                # Set img_rho_t based on whether we're doing a single pair or two pairs of comparisons
                if surface_count==2:
                    img_rho_t = (img - img_rhoMinus) / 2
                else:
                    img_rho_t = (img_rhoPlus - img_rhoMinus) / 2

                if DFD_model == 'FocalSplit':
                    ZMap = getDepthMap_FS(img, img_rho_t, params)
                    ConfidenceMap = getConfidenceMapByIrho(img_rho_t)

                if DFD_model == 'FocalTrack':
                    ZMap, ConfidenceMap, _  = getDepthMap_FT(img, img_rho_t, params)

                filtered_ZMap = filterResultByConfidence(
                    ZMap, ConfidenceMap, ground_distance_m, confidenceLevel
                )

                filtered_ZMap_95 = filterResultByConfidence(
                    ZMap, ConfidenceMap, ground_distance_m, 0.95
                )

                Ztrue = np.full(imageArray.shape, depth)
                error.append(np.nanmean(np.abs(filtered_ZMap - Ztrue)))
                error_95.append(np.nanmean(np.abs(filtered_ZMap_95 - Ztrue)))

            errors.append(error)
            errors_95.append(error_95)

        with open(pkl_output, "wb") as f:
            pickle.dump([errors_95, errors], f)
            f.close()

    with open(pkl_output, "rb") as f:
        errors_95, errors = pickle.load(f)
        f.close()

    # Export MAEs for external plotting and analyses
    if export_MAEs:
    ground_depths_for_csv_m = ['Ground_depth(m)']+[x for x in ground_distance_m]
    ground_degs_for_csv_deg = ['Viewing_Angle(deg)']+[x for x in ground_deg]
    errors100_for_csv_m = ['MAE(m)'] + list(np.nanmean(errors, axis=(0)))

    rows_for_csv = zip(ground_degs_for_csv_deg,ground_depths_for_csv_m,errors100_for_csv_m)

    with open(pkl_output.replace('pkl','csv'), 'w', newline='') as csvfile:
        writer = csv.writer(f'reports/MAE_outputs/{csvfile}')
        # Write all rows at once
        writer.writerows(rows_for_csv)
    print(f"Successfully exported errors to reports/MAE_outputs/{pkl_output}.csv.")


    # Generate MAE plots
    if plot_indiv:
        # Plot at 3 different zoom levels
        zoom_levels = [5, 208, 413]
        for zoom_level_idx in zoom_levels:
            fig = plt.figure(figsize=(24, 24), dpi=50)
            fig.suptitle(f"Avg Tier I DFD Errors based on {model_str}\nSensorDistance: {params['retinalDistance']}, fieldCurvatureDistanceAt0: {params['fieldCurvatureDistance']}\n focusingDistanceAt0: {params['focusingDistance_tag']}, PixelPitch: {PIXEL_PITCH},\nLatest Rho: {params['rho']}, Latest DeltaRho: {params['Delta_rho']}")
            ax = fig.add_subplot(1, 1, 1)

            error_vals = list(np.nanmean(errors, axis=(0)))
            dist_associated_with_minerror = ground_distance_m[error_vals.index(min(error_vals))]

            ### SET UNITS 
            unit_list = [['m', 1], ['cm', 100], ['mm', 1000]]
            depth_idx = 1 # 0=m, 1=cm, 2=mm -- # also 0 for deg
            depth_mul_factor = (unit_list[depth_idx])[1]
            depth_unit = (unit_list[depth_idx])[0]

            error_idx = 2 # 0=m, 1=cm, 2=mm
            error_mul_factor = (unit_list[error_idx])[1]
            error_unit = (unit_list[error_idx])[0]

            ax.set_xlabel(f"Depth ({depth_unit})")
            ax.set_ylabel(f"MAE ({error_unit})")
            # Easy way to subset data to exclude far distances in plots
            slice_at = zoom_level_idx
            
            if slice_at >= 60:
                ax.set_ylim(0, 100)

            ground_deg = [90.0 - x for x in theta_deg]


            with open(pkl_output.replace('pkl','csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write all rows at once
                writer.writerows(rows_for_csv)
            print(f"Successfully exported lists to {pkl_output}.csv.")


            if plot_dist: # Plot with depth as x-axis
                ax.plot(
                    [x*depth_mul_factor for x in ground_distance_m[slice_at:]],
                    np.nanmean(errors, axis=(0))[slice_at:] * error_mul_factor,
                    label="Zmap_100Conf",
                    linewidth=1,
                    color="blue",
                )
                ax.plot(
                    [x*depth_mul_factor for x in ground_distance_m[slice_at:]],
                    [x*error_mul_factor for x in ground_distance_m[slice_at:]],
                    linewidth=3,
                    linestyle="--",
                    color="black",
                )

            else: # Plot with viewing angle as x-axis
                slice_at = 0
                ax.set_xlabel("Viewing pitch (deg)")
                ax.invert_xaxis()

                ax.plot(
                    [x for x in ground_deg[slice_at:]],
                    np.nanmean(errors, axis=(0))[slice_at:] * error_mul_factor,
                    label="Zmap_100Conf",
                    linewidth=1,
                    color="purple",
                )

                ax.plot(
                    [x for x in ground_deg[slice_at:]],
                    [x*error_mul_factor for x in ground_deg[slice_at:]],
                    linewidth=3,
                    linestyle="--",
                    color="black",
                )
            
            ax.grid()
            ax.legend()
            fig.tight_layout()

            overwrite_plots = True
            if save_plot:
                # Write a function that will automatically determine file name
                # Make sure that the nested folder structure is created if the infrastructure is not there
                prefix = ''
                if not fieldCurvature:
                    prefix = 'noCurvature_' + prefix
                prefix = prefix + f'{DFD_model}'
                if surface_count==3:
                    prefix = prefix + '_3surface'
                elif surface_count==2:
                    prefix = prefix + '_2surface'

                prefix = prefix + f'_{initial_focusingDistance_str}'

                if slice_at == 5:
                    prefix = prefix + f'_zoomedout'
                    subfolder = 'zoomedout'
                    prefix = f'{subfolder}/{prefix}'
                if slice_at == 208:
                    prefix = prefix + f'_midzoomed'
                    subfolder = 'midzoom'
                    prefix = f'{subfolder}/{prefix}'
                if slice_at == 413:
                    prefix = prefix + f'_zoomed'
                output_name = f'/reports/MAE_{initial_focusingDistance_str}/{prefix}.png'
                output_path = Path(output_name)
                # Create the parent directories for the file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if (not output_path.exists()) or (overwrite_plots):
                    print(f'Saved plot to: {output_path}')
                    plt.savefig(output_name)
                else:
                    print(f'Did not save: {output_path} since this image existed already ')
            else:
                plt.show()
    return errors, errors_95

def main():
    ### This script has been parameterized to best emulate a Habronattus pyrrithrix jumping spider's
    ### visual system in order to evalute how effective they are at extracting depth information
    ### from defocus (DFDD)
    
    ### The animal's DfDD capabilities are estimated using modified versions of modern DfDD algorithms

    ### The in-axis focusing distance of the animal can be manually initialized here.
    ### Our ophthalmoscopy data indicates that the principal eyes of H. pyrrithrix are emmetropic
    ### Thus we will be setting focusing distance to infinity for realistic simulations.
    focusingDistance_distance_idx = 5 #0=3cm, 1=10cm, 2=30cm, 3=1m, 4=5m, 5=inf
    focusingDistance_list = [0.03, 0.1, 0.3, 1, 5, np.inf]
    focusingDistance_list_tags = ["3cm", "10cm", "30cm", "1m" ,"5m", 'inf']
    focusingDistance = focusingDistance_list[focusingDistance_distance_idx]
    focusingDistance_tag = focusingDistance_list_tags[focusingDistance_distance_idx]

    retinalDistance = 0.00068942+0.000037 # Lens to Retina distance, in m.
                                    # Determined for Habronattus pyrrithrix from x-ray imaging -- average lens-pit-distance 689.42 micron
                                    # Since the x-ray data only gives axial length to pit. We added the pit-to-tier i distance (~37micron) from Zurek et al. 2015 histology.

    retinalGapsize = 1.1e-5  # Distance Between Tier I and Tier II from Zurek et al. 2015 histology.
                             # There is a range of gap sizes present between the tier based on the position along the retina
                             # It ranges from roughly 7 micron to 15 mirons, for the middle of tier i, or the average distances
                             # it would be 11 microns

    OVERWRITE=False ## Useful for running code when discarding previous .pkl output 

    params = {
        "h":0.0025, # The height of the spider's eye, 2.5 mm 
        "Sigma": 0.00039, # 390 microns. Aperture radius in m.
        "PSF_shape": "pillbox", 
        "focusingDistance": focusingDistance,
        "focusingDistance_tag": focusingDistance_tag,
        "rho": 1/retinalDistance + 1/focusingDistance, ## Optical power at rest
        "Delta_rho": (1/(retinalDistance-retinalGapsize) + 1/focusingDistance) - (1/retinalDistance + 1/focusingDistance), # Optical power differential between Tier i and Tier ii of the retina
        "retinalDistance": retinalDistance, 
        "retinalGapsize": retinalGapsize,
        "fieldCurvatureDistance": retinalDistance,
        "kernelSize": 5,
    }

    ### Run DfDD based depth estimations on habitat and texture images under given visual parameters, using the following algorithms

    ### Modified FocalSplit
    ### Firstly, the original FocalSplit implementation varies both Aperture AND Sensor Position between images.
    ### - We modified FocalSplit to only consider changes in sensor position, keeping aperture the same across images.
    ### Secondly, the original FocalSplit algorithm compares focus levels across 3 images.
    ### - We modified it to compare two images reflective of tier 1 and tier 2 of the retina of Habronattus pyrrithrix jumping spiders. 

    ### Depth errors associated with a system that does not exhibit changes in focus arising from Petzval field curvature
    FS_2surface_noCurvature_errors, FS_2surface_noCurvature_errors_95 = getDfDDErrors_multiMethod(params, 'FocalSplit', False, 2,overwrite=OVERWRITE, save_plot=True)
    ### Depth errors associated realistic spider system that exhibits focal shifts resulting from field-based abberations
    FS_2surface_errors, FS_2surface_errors_95 = getDfDDErrors_multiMethod(params, 'FocalSplit', True, 2,overwrite=OVERWRITE, save_plot=True)


    ### Alternate implementations:

    ### Constant aperture Focal Split comparing 3 images
    # FS_3surface_errors, FS_3surface_errors_95 = getDfDDErrors_multiMethod(params, 'FocalSplit', True, 3,overwrite=OVERWRITE, save_plot=True)
    # FS_3surface_noCurvature_errors, FS_3surface_noCurvature_errors_95 = getDfDDErrors_multiMethod(params, 'FocalSplit', False, 3,overwrite=OVERWRITE, save_plot=True)

    ### FocalTrack with 2 and 3 images
    # FT_2surface_errors, FT_2surface_errors_95 = getDfDDErrors_multiMethod(params, 'FocalTrack', True, 2,overwrite=OVERWRITE, save_plot=True)
    # FT_3surface_noCurvature_errors, FT_3surface_noCurvature_errors_95 = getDfDDErrors_multiMethod(params, 'FocalTrack', False, 3,overwrite=OVERWRITE, save_plot=True)
    # FT_3surface_errors, FT_3surface_errors_95 = getDfDDErrors_multiMethod(params, 'FocalTrack', True, 3,overwrite=OVERWRITE, save_plot=True)
    return

if __name__ == "__main__":
    main()