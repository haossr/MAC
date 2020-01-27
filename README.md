# MAC (Model Agnostic Compressor) 
[![CodeFactor](https://www.codefactor.io/repository/github/haossr/mac/badge)](https://www.codefactor.io/repository/github/haossr/mac)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
This is a starter repo providing skeleton of ML projects for image classification, image segmentation and image generation. Please fork this repo to get started and share your brilliant ideas through pull request. 


## Quick start
### Train and save a model 
`python main.py train --<hyperparameter> value`

### Test existing model 
`python main.py test --checkpoint_path <path to checkpoint>`

## Useful links 
- [Negative sampling (Google Map API)](https://github.com/stanfordmlgroup/old-starter/blob/master/preprocess/get_negatives.py)
- [Example of dataset implementation: USGS dataset](https://github.com/stanfordmlgroup/old-starter/blob/master/data/usgs_dataset.py)
- [Documentation for Fire](https://github.com/google/python-fire/blob/master/docs/guide.md)
- [Documentation for Pytorch Lighning](https://williamfalcon.github.io/pytorch-lightning/)

## Troubleshooting Notes
- Inplace operations in PyTorch is not supported in PyTorch Lightning distributed mode. Please just use non-inplace operations instead. 

--- 
Maintainers: [@Hao](mailto:haosheng@cs.stanford.edu)
 
