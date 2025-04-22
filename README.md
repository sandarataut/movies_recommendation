# movies_recommendation_system

This project not only serves as a practical application of my study in Master of Computer & Information Science but also addresses the real-world need for a personalized entertainment experience in the digital age.

# dataset

- [MovieLens Dataset link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data)

# How to run?

### Steps:

### STEP 01- clone the repository

```bash
git clone https://github.com/sandarataut/movies_recommendation.git
```

### STEP 02- open the repository

```bash
cd movies_recommendation
```

### STEP 03- check existing conda environment after opening the repository

```bash
conda env list
```

### STEP 04- create and activate a conda environment

```bash
conda create -n movies_recommendation_system python=3.12.8 -y
```

```bash
conda activate movies_recommendation_system
```

### STEP 05- install kernel package and create a new kernel

```bash
conda install ipykernel
```

```bash
python -m ipykernel install --user --name movies_recommendation_system --display-name "movies_recommendation_system"
```

### STEP 06- install the necessary packages / libriaries

```bash
pip install -r requirements.txt
```

### STEP 07- run this file to generate the models

```bash
01_data_prep.ipynb
```

### STEP 08- run the project in localhost

```bash
streamlit run app.py
```

# What I have done?

### Steps:

1. created a github repo: git@github.com:sandarataut/movies_recommendation.git
2. cloned this repo to my localhost
3. showing existing conda environments (default is "base")
   - conda env list
4. created a new conda environment
   - conda create -n xxx
5. activated existing environment
   - conda activate xxx
6. installed kernel package within existing environment for enabling switch the kernel in VS Code or Jupyter Notebook
   - conda install ipykernel
7. created a new kernel within existing environment
   - python -m ipykernel install --user --name xxx --display-name "xxx"
8. installed necessary packages / libriaries
   - pip install -r requirements.txt
     [OR]
   - pip install joblib
   - pip install matplotlib
   - pip install nltk
   - pip install numpy
   - pip install pandas
   - pip install requests
   - pip install scikit-learn
   - pip install sentence-transformers
   - pip install scipy
   - pip install streamlit
   - pip install tokenizers
   - pip install torch
   - pip install transformers
9. run the project in localhost
   - streamlit run app.py
