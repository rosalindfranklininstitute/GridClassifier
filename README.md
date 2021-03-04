# GridClassifier
To classify grids from their intial images

Example of use:

look at the help
```
python .\src\GridMapper.py --help
```

train a model
```
python .\src\GridMapper.py .\example_data\Scans\Grid1_back -p .\example_data\Properties\Grid1_back\PROP0001.jpg
```

Create a prediction from the training.
```
python .\src\GridMapper.py .\example_data\Scans\Grid1_back -o test.jpg
```

