
UNet is a convolutional neural network (CNN) architecture primarily designed for biomedical image segmentation, particularly for the task of semantic segmentation in medical imaging. It was introduced by Ronneberger et al. in 2015 \cite{ronneberger2015u}. The distinguishing feature of UNet is its symmetric U-shaped architecture, which consists of a contracting path for capturing context and a symmetric expanding path for precise localization. The contracting path consists of a series of convolutional and pooling layers to extract hierarchical features, while the expanding path uses transposed convolutions for upsampling and reconstructing the segmented image. Skip connections between corresponding layers in the contracting and expanding paths enable the network to recover spatial details effectively, making it particularly adept at handling tasks where precise localization is crucial. UNet's effectiveness stems from its ability to accurately segment images while handling limited training data, making it versatile for various image-related tasks beyond its initial biomedical applications, which we decided to utilize for finding out if we are able to recover image contents from the phase spectrum with its use.

**Results**

The model was trained on a dataset consisting of 4000 images, with the remaining 1000 images left for testing. During training, we tried various configurations of the model, including different numbers of convolutional layers and hyperparameters. After experimentation, the configuration with a learning rate of 0.00001, a batch size of 32, and training for 15 epochs was selected.

Despite achieving a low average Mean Squared Error (MSE) of  0.0849, the visual quality of the model's output did not meet expectations.The low MSE is most likely caused by the normalization. Normalization is a common preprocessing step that can help stabilize and accelerate training but may not fully capture the nuances of the image data, which seems to be the case here.

In a previous iteration of training, a slightly different model configuration was used, with a learning rate of 0.001, trained for 10 epochs, and without batch normalization layers and the pixel values were not normalized. In that case, the MSE was significantly higher at 18389.71. 


| without normalization      |with normalization, 5 epochs | with normalization, 15 epochs |
| :---        |    :----:   |          ---: |
| 18389.71      | 0.183497    | 0.0849   |

Results of the chosen model:
![result 1](bocian_likesu/results/result_norm_15epochs.png)
Results of the same model configuration with 5 epochs training:
![result 2](bocian_likesu/results/result__norm_5epochs.png)
Results of the second mentioned model configuration, without normalization:
![result 3](bocian_likesu/results/result_wo_norm.png)


We can conclude that the chosen model, despite not giving truly satisfactory results, is better than than the one without normalization and the one with only 5 epochs training.

@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18},


