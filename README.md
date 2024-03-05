# PLEGDF-FGLSCF
The PyTorch implementation for the paper ' Depression Intensity Recognition Based on Perceptually Locally-enhanced Global Depression Features and Fused Global-local Semantic Correlation Features on Faces'.

Qiang Sun<sup>1</sup>,Zheng Li<sup>2</sup>, Lang He<sup>3</sup>
1.	Department of Communication Engineering, School of Automation and Information Engineering, Xi’an University of Technology, Xi’an 710048, China
2.	Department of Electronic Engineering, School of Automation and Information Engineering, Xi’an University of Technology, Xi’an 710048, China
3. School of Computer Science and Technology, Xi'an University of Posts and Telecommunications, Xi’an 710121, China
# Overview

Abstract: 
For automatic recognition of the depression intensity in patients, the existing deep learning based methods typically face two main challenges: (1) It is difficult for deep models to effectively capture the global context information relevant to the level of depression intensity from facial expressions, and (2) the semantic consistency between the global semantic information and the local one associated with depression intensity is often ignored. One new deep neural network for recognizing the severity of depressive symptoms, by combining the Perceptually Locally-enhanced Global Depression Features and the Fused Global-local Semantic Correlation Features (PLEGDF-FGLSCF), is proposed in this paper. Firstly, the PLEGDF module for the extraction of global depression features with local perceptual enhancement, is designed to extract the semantic correlations among local facial regions, to promote the interactions between depression-relevant information in different local regions, and thus to enhance the expressiveness of the global depression features driven by the local ones. Secondly, in order to achieve full integration of global and local semantic features related to depression severity, we propose the FGLSCF module, aiming to capture the correlation of global and local semantic information and thus to ensure the semantic consistency in describing the depression intensity by means of global and local semantic features. On the AVEC2013 and AVEC2014 datasets, the PLEGDF-FGLSCF model achieved recognition results in terms of the Root Mean Square Error (RMSE) and the Mean Absolute Error (MAE) with the values of 7.75/5.96 and 7.49/5.99, respectively, demonstrating its superiority to most existing benchmark methods, verifying the rationality and effectiveness of our approach.

# Dependencies
+ Python 3.6
+ PyTorch 1.7.1
+ torchvision 0.8.2

# Data availability
+ AVEC2013: http://avec2013-db.sspnet.eu/
+ AVEC2014: http://avec2014-db.sspnet.eu/


# Contact
If you have any questions, please feel free to reach me out at qsun@xaut.edu.cn and 2200320116@stu.xaut.edu.cn.
# Citation
If this work is useful to you, please cite:

> Sun, Q., Li, Z., He, L.: Depression Intensity Recognition Based on Perceptually Locally-enhanced Global Depression Features and Fused Global-local Semantic Correlation Features on Faces[J]. Journal of Electronics & Information Technology.  doi: 10.11999/JEIT231330.
