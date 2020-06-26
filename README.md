# Image-Captioning-Using-Attention-Modelling-and-BERT-Embeddings-CS-682-

## Introduction
We present a template that produces explanations of the activities or the objects of the image in the natural lan- guage. Our approach uses image data-set and word definitions to learn about the inter-model relationship between language and visual information. Our model of alignment is based on a novel combination of Convolution Neural Net- works over image regions, Recurrent Neural Networks over sentences, and a structured goal that aligns the two modal- ities with an embedded BERT. We will demonstrate that our model generates state-of-the-art results in MS-COCO data-set retrieval experiments. Then we show that the descriptions produced significantly outperform the baselines of retrieval on complete images.

## Conclusion
We propose addition to the baseline model of CNN and LSTM approach of image captioning using attention models that provides state of the art performance on the MS COCO dataset using BLEU metric. The BERT approach surpasses the MS COCO validation scores of the Attention model while being trained on fewer epochs with the same hyper-parameters. Our experiments outline the importance of word embedding in the processing of the language of na- ture and contextualizing word meanings, and also offer a new way of integrating BERT with already developed models to improve their performance.
Future ideas would include to train a new model with BERT large as apposed to the BERT base which was used here. A good idea will be to use image data-set with occlusions with a BERT-large embedding and check the performance. Although we could not train the model as we would have liked to, for large enough epochs and better hyper-parameter tuning given the heavy time constraints and a limited GPU access, the learning included in-detail understanding of the encoder-decoder architecture, attention models and the BERT embedding technique used in natural language processing. Other ideas would be to utilize beam search validation, train the models until the training loss converges, i.e, for a higher number of epochs.

## Installations

- Numpy
- PyTorch
- Matplotlob

Final Project for CS682(University of Massachusetts Amherst). 
Contains Code, Results and the Project Report.
