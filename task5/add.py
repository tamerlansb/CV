filenames = list(train_gt.keys())
data = np.zeros(100, 224, 224,3))

test = np.array([i for i in range(2500) if (i+1)%25==0 ]) 

for i in test:
        filename = join(train_img_dir, filenames[i])
        pict = imread(filename)
        if len(pict.shape)==2:
            temp_pict = np.zeros((pict.shape[0],pict.shape[1],3))
            temp_pict[...,0] = pict
            temp_pict[...,1] = pict
            temp_pict[...,2] = pict
            pict = temp_pict
        pict = imresize(pict , (224, 224), interp = 'bicubic')
        data[i, :] = pict
data = preprocess_input(data)


labels = np.zeros(100, 50))
j = 0
for i in test:
    labels[j][i] = 1
    j = j + 1


data = data[test]
labels = lab


