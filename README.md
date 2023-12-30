<h1 style="text-align: center;">Transformers compete on imbalanced product classification data</h1>

The dataset can can be found in this link. <a href="https://drive.google.com/drive/folders/165k0tlrmcF28n1fz1DVSwXXXAdt5ONZJ?usp=sharing">[link]</a>

<h3>Steps taken in this project:</h3>

1. The dataset is explore and pre-processed. <a href="./Eda_and_preprocessing.ipynb">[EDA and preprocessing]</a>

2. The pre-processed dataset is tokenized 
using **rust** based batchwise tokenization and saved as pickle files. <a href="./tokenization.ipynb">[tokenization]</a>

3. Each transformer is trained using its respective tokensized dataset, the metrics are logged, and then visualized as shown below. <a href="./Training.ipynb">[training]</a> : <br><br>


<p align="center">
  <img src="./result_visualizations/benchmarking_transformers.png" alt="Girl in a jacket" width="850" height="550">
</p>
<br>
4. The top performer is retrained using the focal loss, and the improved is shown as below.<br><br><br>
<p align="center">
  <img src="./result_visualizations/focal_vs_default.png" alt="Girl in a jacket" width="650" height="350">
</p>

______

