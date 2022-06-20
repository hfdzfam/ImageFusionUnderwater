# UnderwaterImageFusion

## Project Description
Underwater image enhancement system with fusion using only single image as input.

## Background
Capturing clear images in underwater environments is an important issue of ocean engineering. The effectiveness of applications such as underwater navigational monitoring and environment evaluation depend on the quality of underwater images. Capturing clear images underwater is challenging, mostly due to haze caused by color scatter in addition to color cast from varying light attenuation in different wavelengths. Color scatter and color cast result in blurred subjects and lowered contrast in underwater images. Capturing clear images in underwater environment is challenging due to haze caused by color scatter in addition to color cast from varying light attenuation in different wavelengths. Color scatter and color cast result in blurred subjects and lowered contrast in underwater images. 

## Method
This enhancing strategy consists of three main steps: 
1. Inputs assignment (derivation of the inputs from the original underwater image);
2. Defining weight measures and multiscale;
3. Fusion of the inputs and weight measures.

![Method Picture](https://user-images.githubusercontent.com/77296025/174542606-7db62536-92d1-4fa3-b9c9-b6a11444304e.png)

The degraded image is firstly white balanced in order to remove the color cast while producing a natural appearance of the sub-sea images. This partially restored version is then further enhanced by suppressing some of the undesired noise. The second input is derived from this filtered version in order to render the details in the entire intensity range.

The general idea of image fusion is that the processed result, combines several input images by preserving only the most significant features of them.

## Getting Started
### Dependencies
1. Visual Studio
2. OpenCV3.x
3. OpenCV Contrib

### Installation
Copy the FusionMain.cpp code to your main visual studio project main program.
Or copy the whole visual studio project to your workspace,open with visual studio
(This project created using VS2013)

## Result
### Input

### Output

### Reference
1. Cosmin Ancuti, Codruta Orniana Ancuti, Tom Haber and Philippe Bekaert, Enhancing Underwater Images and Videos by Fusion, Hasselt University - tUL -IBBT, EDM, Belgium
