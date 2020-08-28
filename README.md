# [SIGGRAPH 2020] One Shot 3D Photography

This is the code for the ``Tiefenrausch'' depth estimation method, described in our paper
**One Shot 3D Photography**
in SIGGRAPH 2020.

It reproduces the row in Table 1 labeled "Tiefenrausch (AS + quant)", and can be used for evaluation.

To achieve better quality, i.e., as in the version we use in the Facebook App, it needs to be trained with a more varied dataset such as in the last row in Table 1.

This method produces depth maps only. If you want to create 3D photos, you can use the Facebook app, or if you're looking for an OSS implementation, you can use this code https://github.com/vt-vl-lab/3d-photo-inpainting

Please find more details on our project page:
https://facebookresearch.github.io/one_shot_3d_photography/

## Prerequisites:

```
conda create -yn one_shot python=3.7
conda activate one_shot
conda install --file requirements.txt -c conda-forge -c pytorch
```

## Process a single image

```
python cli.py --src_file ./input/pumpkin.jpg --out_file ./output/pumpkin.npy --vis_file ./output/pumpkin.png
```

## Process a directory

```
python cli.py --src_dir ./input/ --out_dir ./output/ --vis_dir ./output/
```

## Citation

If you use our code, consider citing our paper:
```
@article{Kopf-OneShot-2020,
  author    = {Johannes Kopf and Kevin Matzen and Suhib Alsisan, Ocean Quigley and Francis Ge and Yangming Chong and Josh Patterson and Jan-Michael Frahm and Shu Wu and Matthew Yu and Peizhao Zhang and Zijian He and Peter Vajda and Ayush Saraf and Michael Cohen},
  title     = {One Shot 3D Photography},
  booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
  publisher = {ACM},
  volume = {39},
  number = {4},
  year = {2020}
}
```

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.
