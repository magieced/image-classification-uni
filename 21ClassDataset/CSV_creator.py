import os 
# Citations

 # Citation of The Oxford-IIIT Pet Dataset
    # Dataset from Kaggle: https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset?resource=download
    # Paper
        # Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, 
        # and C. V. Jawahar. Cats and dogs. 
        # In IEEE Conference on Computer Vision and Pattern Recognition, 2012.
    # Funding: The UK India Education and Research Initiative and European Reasearch Council Grant VisRec

# Citation of The ImImageNet Large Scale Visual Recognition Challenge (ILSVRC) / ImageNet-1k
    # Datasets from Kaggle
        # ImageNet-1k-0: https://www.kaggle.com/datasets/sautkin/imagenet1k0
        # ImageNet-1k-2: https://www.kaggle.com/datasets/sautkin/imagenet1k2
    # Paper of ImageNet-1k
        # Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, 
        # Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, 
        # Aditya Khosla, Michael Bernstein, Alexander C. Berg, 
        # and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. 
        # International Journal of Computer Vision (IJCV), 115 (3):211–252, 2015.
    # Paper of ImageNet
        # Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
        # and Li Fei-Fei. Imagenet: A large-scale hierarchical image
        # database. In 2009 IEEE conference on computer vision and
        # pattern recognition, pages 248–255. IEEE, 2009.

# Citation Cat Breeds Dataset
    # Dataset from Kaggle
        # Cat Breeds Dataset: https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset
        # Dataset made available by:
            # PetFinder API: https://www.petfinder.com
            # aschlg: petpy: https://github.com/aschleg/petpy
            # User (ma7555): multiprocess image downloader: https://github.com/ma7555/petpy/blob/new/notebooks/03-Download%20Pure%20Breeds%20Cat%20Images%20with%20petpy%20for%20Deep%20Neural%20Network%20training%20-%20multiprocessing.ipynb

# Citation Stanford Dogs Datasets (uses images from ImageNet)
    # Dataset from Kaggle
        # Standford Dogs Dataset: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
    # Paper of Stanford Dogs Dataset (Primary source)
        # Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao 
        # and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. 
        # First Workshop on Fine-Grained Visual Categorization (FGVC), 
        # IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.
    # Paper of Stanford Dogs Dataset (Secondary source)
        # J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, 
        # ImageNet: A Large-Scale Hierarchical Image Database. 
        # IEEE Computer Vision and Pattern Recognition (CVPR), 2009.

# To note: per class there are 199 images from the above cited datasets
        # amounting to a total image count of 4179 images

list_of_classes = ["REJECT", "Abyssinian", "Bengal", "Birman", "Bombay", 
                   "British_Shorthair", "Maine_Coon", "Ragdoll", 
                   "Sphynx", "Tabby", "Tiger_Cat", "Beagle", 
                   "Pug", "Boxer", "Shiba_Inu", "Samoyed", 
                   "Golden_Retriever", "German_Shepherd", "Siberian_Husky", 
                   "Dalmatian", "Rottweiler"]

csv_file_path = "labels_21ClassDataset.csv"
with open(csv_file_path, "w", newline="") as csv_file:
    csv_file.write("filename,label\n")
    label_counter = -1
    for elem in list_of_classes: 
        element = os.listdir(elem)
        
        for jpg in element: 
            if jpg == ".DS_Store":
                continue
            csv_file.write(elem + "/" + jpg + "," + str(label_counter) + "\n")
        label_counter += 1  
print("labels_21ClassDataset created")
