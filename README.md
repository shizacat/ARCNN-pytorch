# AR-CNN, Fast AR-CNN

This repository is implementation of the "Deep Convolution Networks for Compression Artifacts Reduction".

In contrast with original paper, It use RGB channels instead of luminance channel in YCbCr space and smaller(16) batch size.

Papers:
- [arXiv:1504.06993v1 27 Apr 2015](https://arxiv.org/pdf/1504.06993.pdf): Compression Artifacts Reduction by a Deep Convolutional Network. ARCNN
- [arXiv:1608.02778v1 09 Aug 2016](https://arxiv.org/pdf/1608.02778.pdf): Deep Convolution Networks for Compression Artifacts Reduction. Fast ARCNN

## Requirements
- PyTorch
- tqdm
- Numpy
- Pillow

## Usages

### Train

When training begins, the model weights will be saved every epoch.

Data for training, the folder has to content two directories: train and val,
which it store the image in format png.

```bash
python train.py --arch "ARCNN" \     # ARCNN, FastARCNN
                --data_folder "" \
                --outputs_dir "" \
                --save_result 10 \
                --batch_size 16 \
                --num_epochs 20 \
                --lr 5e-4 \
                --seed 123 \
                --model ""
```

### Test

Output image will be stored in same folder under the name "out_<name>.png".

```bash
python process.py --arch "ARCNN" \     # ARCNN, FastARCNN
                  --model "" \
                  --image ""
```

## Conclusion

Source: [yjn870/ARCNN-pytorch](https://github.com/yjn870/ARCNN-pytorch)
