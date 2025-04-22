# VAE-GAN-based-model-for-image-generation-and-denoising
This practical project aims to develop a deep learning model capable of generating synthetic images and performing image denoising. This practical project aims to develop a deep learning model capable of generating synthetic images and performing image denoising. To achieve this, we propose using a known architecture, which combines the representational strength of a Variational Autoencoder (VAE) with the image generation quality of a Generative Adversarial Network (GAN), called VAE-GAN [1]. The model comprises three key components (see Fig. 1a): (i) an encoder, (ii) a decoder/generator, and (iii) a discriminator. The encoder processes noisy input images and maps them to a meaningful latent representation. The generator then reconstructs images from this latent space. Depending on the input, the latent vector can either be sampled randomly—for synthetic image generation (see Fig. 1b)—or obtained from the encoder to perform image denoising (see Fig. 1c). The discriminator is used during training to distinguish between real and generated images, encouraging the generator to produce realistic outputs.

![training_page-0001](https://github.com/user-attachments/assets/634038b2-a30a-4156-8125-25d9efe656e7)

## Methodology

To carry out this project, the VAE-GAN model will be implemented using PyTorch [1]. Each component consists of a Convolutional Neural Network, with its configuration adapted to a specific task (upsampling for the decoder/generator, downsampling for the encoder and discriminator). The GAN loss (1) uses a binary cross-entropy loss with respect to Discriminator/Generator output, while the VAE loss (3) uses log-likelihood (expressed in GAN discriminator) and a regularization term based on Kullback-Leibler divergence. For the data sample x, with latent representation z, the mentioned losses are given by:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?\dpi{200}\bg{white}\begin{matrix*}\color{white}{000000000000000000000000000000000000000000000}&\\\mathcal{L}=\mathcal{L}_{\text{Dis}_l}+\mathcal{L}_{\text{prior}}+\mathcal{L}_{\text{GAN}}\\\mathcal{L}_{\text{GAN}}=\log(\text{Dis}(x))+\log(1-\text{Dis}(\text{Gen}(z)))&(1)\color{white}{00}\\\mathcal{L}_{\text{Dis}_l}=-\mathbb{E}_{q(z|x)}\left[\log{p(\text{Dis}_l(x)|z)}\right]&(2)\color{white}{00}\\\mathcal{L}_{\text{prior}}=\text{D}_{\text{KL}}(q(z|x)\|p(z))&(3)\color{white}{00}\\\color{white}{000000000000000000000000000000000000000000000}\end{matrix*}"/>
</p>

Where (2) is the representation at the l-th hidden layer of the discriminator. In the original proposal, Larsen et al. [2] present an algorithm for training these models: first, the input sample/batch is encoded and decoded by the first two components, and the loss between them is computed; samples from a prior p(z) are then passed through the decoder to generate an artificial input; finally, the original input, encoded-decoded input, and artificial input are passed to the discriminator to calculate the GAN loss, and parameters for the three components are updated according to the gradients. Initially, a well-established dataset will be used to assess the validity of the implementation and the impact of several hyperparameters on overall performance, namely convolutional layer size across the different components. After this, a VAE-GAN model will be trained on datasets relevant to other applications.

## Datasets

Our aim in this work is to have a model that is capable of generating and denoising images within a specific domain. Initially, the networks will be trained for image generation and denoising tasks, specifically using facial images. For this purpose, the CelebA dataset [3], a widely recognized benchmark, will serve as a reference for guiding the architectural design and evaluating the model’s generative and denoising capabilities. Following this initial phase, the model will be extended to address three distinct application domains: sonar imagery, EEG signals, and LiDAR ranging.

## Evaluation Metrics

To evaluate image generation quality, two metrics are intended to be used: Inception Score (IS) [4] and Fréchet Inception Distance (FID) [5], both based on InceptionV3 features. IS evaluates image quality/diversity via KL divergence; FID compares feature distributions under a Gaussian assumption. Since these ignore overfitting [6], Precision, Recall, and F1-score are also used, computed via manifold distances [7]. For image denoising, Peak Signal to Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Mean Squared Error (MSE) are the intended metrics to use [8, 9].

## References
[1] Anders Boesen Lindbo Larsen et al. Autoencoding beyond pixels using a learned similarity metric. 2015. doi: 10.48550/ARXIV.1512.09300. url: https://arxiv.org/abs/1512.09300

[2] Adam Paszke et al. ‘PyTorch: An Imperative Style, High-Performance Deep Learning Library’. In: Advances in Neural Information Processing Systems 32. Curran Associates, Inc., 2019, pp. 8024–8035. url: https://arxiv.org/abs/1912.01703

[3] Ziwei Liu et al. ‘Deep Learning Face Attributes in the Wild’. In: Proceedings of International Conference on Computer Vision (ICCV). Dec. 2015. url: https://arxiv.org/abs/1411.7766

[4] Tim Salimans et al. ‘Improved techniques for training gans’. In: Advances in neural information processing systems 29 (2016). url: https://arxiv.org/abs/1606.03498

[5] Martin Heusel et al. ‘GANs trained by a two time-scale update rule converge to a local nash equilibrium’. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. Red Hook, NY, USA: Curran Associates Inc., Dec. 2017, pp. 6629–6640. url: https://arxiv.org/abs/1706.08500

[6] Zeeshan Ahmad et al. ‘Understanding GANs: fundamentals, variants, training challenges, applications, and open problems’. In: Multimedia Tools and Applications (May 2024). doi: 10.1007/s11042-024-19361-y.

[7] Tuomas Kynkaanniemi et al. ‘Improved precision and recall metric for assessing generative models’. In: Proceedings of the 33rd International Conference on Neural Information Processing Systems. Red Hook, NY, USA: Curran Associates Inc., Dec. 2019, pp. 3927–3936.  url: https://arxiv.org/abs/1904.06991

[8] Rini Smita Thakur et al. ‘Image De-Noising With Machine Learning: A Review’. In: IEEE Access 9 (2021), pp. 93338–93363. doi: 10.1109/ACCESS.2021.3092425. url: https://www.researchgate.net/publication/352806683

[9] Rini Smita Thakur, Ram Narayan Yadav and Lalita Gupta. ‘State-of-art analysis of image denoising methods using convolutional neural networks’. In: IET Image Processing 13.13 (Nov.2019), pp. 2367–2380. doi: 10.1049/iet-ipr.2019.0157.  url: https://www.researchgate.net/publication/335036812
