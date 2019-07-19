# discrete_cosine_transform

Perform discrete cosine transform of image.  
1. Execute the following code to build the environment.  
```python setup.py install```  

2. Save the image in the "image" folder  

3. Please change the following part of dct_fig.py according to the image file to be converted.    

```python:dct_fig.py  
# Assign the number of pixels of the image.
# Example: N = 50 for 50 * 50
N = 50
# Specify range to be rounded to 0
number = 15

# Image loading
image = np.array(Image.open('./image/image3.jpg'))
```

4. Execute dct_fig.py  
```python dct_fig.py```  
