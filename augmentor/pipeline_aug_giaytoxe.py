
import cv2
import numpy as np 
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa 
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug.parameters as iap
import imgaug
from tqdm import tqdm

from custom_augmentation import OverLayMultipleImage, imgaug_to_normal

from custom_augmentation import (LightFlare, OverLayImage, ParallelLight,
                                 SpotLight, imgaug_to_normal, WarpTexture, RandomLine, BlobAndShadow)

def blur_augment(num=1):
    aug = iaa.SomeOf(num, [
        # Normal blur
        iaa.GaussianBlur(sigma=(1.0, 3.0)),
        iaa.AverageBlur(k=(2, 3)),
        # iaa.MedianBlur(k=3),
        iaa.MotionBlur(k=(3, 5)),
        iaa.BilateralBlur(d=(3, 4), sigma_color=(10, 250), sigma_space=(10, 250)),
        # Corrupt pixel
        iaa.AveragePooling((1, 3)),
        iaa.MaxPooling((1, 2)),
        # iaa.JpegCompression(compression=(70,99)),

        iaa.imgcorruptlike.DefocusBlur(severity=1),
        iaa.imgcorruptlike.GlassBlur(severity=1),
        # iaa.imgcorruptlike.MotionBlur(severity=(1, 3)),
        
        iaa.imgcorruptlike.Pixelate(severity=(1,3)),
        # iaa.imgcorruptlike.JpegCompression(severity=(1,5)),

        iaa.BlendAlphaSomeColors(iaa.AveragePooling(2), alpha=[0.0, 1.0], smoothness=0.0)
    ])
    return aug


def noise_augment(num=1):
    aug = iaa.SomeOf(num, [
        iaa.Pepper(0.1),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1*255), per_channel=True),
        iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=True),
        iaa.AdditivePoissonNoise(40, per_channel=True),
        iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
        iaa.Dropout(p=(0, 0.2), per_channel=0.5),
        iaa.ReplaceElementwise(0.1, iap.Normal(128, 0.4*128), per_channel=0.5),
        iaa.imgcorruptlike.SpeckleNoise(severity=1),
    ])
    return aug 


def weather_augment(num=1):
    aug = iaa.SomeOf(num, [
        iaa.imgcorruptlike.Brightness(severity=(1,3)),
        iaa.imgcorruptlike.Saturate(severity=1),
        iaa.pillike.EnhanceSharpness(),
        iaa.imgcorruptlike.Spatter(severity=(1,3)),
        iaa.CoarseDropout(0.02, size_percent=0.01, per_channel=1),
        iaa.imgcorruptlike.Contrast(severity=1),
        iaa.imgcorruptlike.Snow(severity=1),
        iaa.imgcorruptlike.Frost(severity=1),
        iaa.imgcorruptlike.Fog(severity=(1, 3)),
        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
        iaa.MultiplyBrightness((0.8, 1.05)),
        iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((-20, 20)))),
        iaa.ChangeColorTemperature((2000, 40000))
    ])
    return aug

def blend_augment(list_augmenter: tuple=None):
    aug = iaa.OneOf([
        # Blend list
        # iaa.BlendAlphaSimplexNoise(
        #     foreground=list_augmenter,),
        iaa.BlendAlphaVerticalLinearGradient(
            # list_augmenter,
            iaa.Clouds(),
            start_at=(0.0, 1.0), end_at=(0.0, 1.0)
        ),
        iaa.BlendAlphaHorizontalLinearGradient(
            # list_augmenter,
            iaa.Clouds(),
            start_at=(0.0, 1.0), end_at=(0.0, 1.0)
            ),

        iaa.BlendAlphaFrequencyNoise(
            exponent=(-4,4),
            foreground=(
                # list_augmenter
                iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
            ),
            size_px_max=32,
            upscale_method="linear",
            iterations=1,
            sigmoid=False
        ),
        iaa.BlendAlphaSomeColors(iaa.AveragePooling(4), alpha=[0.0, 1.0], smoothness=0.0),
        iaa.BlendAlphaMask(
            iaa.InvertMaskGen(0.35, iaa.VerticalLinearGradientMaskGen()),
            iaa.Clouds()
        ),
        iaa.BlendAlphaSimplexNoise(
            foreground=(
                iaa.EdgeDetect(0.5),
                iaa.Multiply(iap.Choice([0.9, 1.1]), per_channel=True),
            ),
            per_channel=True,
            upscale_method="linear"
        ),
        iaa.BlendAlphaFrequencyNoise(
            exponent=(-4,4),
            foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
            size_px_max=32,
            upscale_method="linear",
            iterations=1,
            sigmoid=False,
            per_channel=True
        ),
        iaa.SimplexNoiseAlpha(
            iaa.Multiply(iap.Choice([0.8, 1.2]), per_channel=True)
        ),
    ])
    return aug


augment_pipeline_1 = iaa.Sequential([
    iaa.Sometimes(0.2, LightFlare()),
    iaa.Sometimes(0.2, ParallelLight()),
    iaa.Sometimes(0.2, SpotLight()),
    iaa.Sometimes(0.2, RandomLine()),
    iaa.Sometimes(0.2, BlobAndShadow()),
    iaa.Sometimes(0.1, WarpTexture()),
    iaa.Sometimes(0.5, 
        iaa.OneOf([
            iaa.Affine(
                scale=(1.0, 1.2),
                rotate=(-20, 20),
                # shear=(-15, 15),
                translate_percent=(-0.2, 0.2),
                cval=255,
                mode='constant',
                fit_output=True
            ),
            iaa.PerspectiveTransform(scale=(0.01, 0.05), mode=ia.ALL ,keep_size=True, 
                            fit_output=True, polygon_recoverer="auto", cval=(0,255)),
            # can not use this augmentaton 
            # iaa.PiecewiseAffine(scale=0.01, nb_rows=4, nb_cols=4),
            
        ])
    ),
    # iaa.Sometimes(0.2, 
    #     iaa.Affine(
    #         rotate=(-100, 100), 
    #         fit_output=True
    #     )),
    iaa.Sometimes(0.3, iaa.OneOf([
        blur_augment(),
        noise_augment(),
    ])),
    # iaa.Sometimes(0.1, weather_augment()),
    iaa.Sometimes(0.05, blend_augment()),
            # blur_augment(), 
            # weather_augment(), 
            # noise_augment()),
    # ),

    # Can not use this augmentaton
    # iaa.Sometimes(0.05, iaa.imgcorruptlike.ElasticTransform(severity=(3,5))), 
    iaa.Sometimes(0.05, iaa.Crop(percent=(0.0, 0.05))),
])

augment_pipeline_2 = iaa.Sequential(
    [
        iaa.Sometimes(0.6, iaa.OneOf(
            [
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                SpotLight(),
                ParallelLight()
            ]
        )),
        iaa.OneOf(
            [
                iaa.Sometimes(0.325, iaa.MotionBlur(k=3)),
                iaa.Sometimes(0.325, iaa.GaussianBlur(sigma=(0.05, 0.1))),
                iaa.Sometimes(0.325, iaa.MedianBlur(k=3))
            ]
        ),
        iaa.Sometimes(0.8, iaa.OneOf([
            iaa.Add(value=(-30, 30)),
            iaa.GammaContrast(gamma=(0.65, 1.15))
        ])),
        iaa.Sometimes(0.65, iaa.AdditiveGaussianNoise(
            scale=(0.01*255, 0.06*255))),
        iaa.OneOf([
            iaa.Sometimes(0.35, iaa.JpegCompression(compression=(10, 30))),
            iaa.Sometimes(0.4, iaa.ElasticTransformation(
                alpha=(0, 0.5), sigma=0.3))
        ])
    ])

augment_pipeline_blur = iaa.Sequential(
    [
       iaa.OneOf(
            [
                iaa.Sometimes(0.5, iaa.MotionBlur(k=(3,5))),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.05, 0.1))),
                iaa.Sometimes(0.5, iaa.MedianBlur(k=(3,5))),
                iaa.Sometimes(0.5, iaa.AverageBlur(k=(3,5)))
            ]
        ), 
    ]
)