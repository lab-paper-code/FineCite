# *FINECite : A novel three-class framework for fine-grained citation context analysis*

We realize this framework by constructing a novel corpus containing 1,056 manually annotated fine-grained citation contexts. Next, we establish baseline models for two important applications in citation context analysis: citation context extraction and citation context classification. Importantly, our experiments demonstrate the positive impact of our finer-grained context definition leading to an increase in performance on both tasks compared to previous approaches.


<p align="center">
  <img src="https://github.com/user-attachments/assets/2096d5f0-91bd-4133-9880-4eda813aa822" width="75%" alt="Comparing_table"/>
</p>


## The Explanation of the FINCite code.
1. data : The storage of datas for our paper
2. corpus_construction : Data processing
3. model_training
    1. seq_tagger.py : model training and evaluation
    2. output : the storage of reults after model running
    3. extract_resulty : Aggregate the results to show which result is the best performance
    4. finecite_scopes_weights.json & finecite_total_weights.json : It is used to train the model.
4. classification_task


## How to use:

1. Create virtual environment:
    ```
    virtualenv venv
    ```

2. Activate virtual environsment
    ```
    source venv/bin/activate
    ```

3. install requirements_finecite.txt
    ```
    pip install -r requirements_finecite.txt
    ```

4. Update requirement.txt
    ```
    pip freeze >requirements_finecite.txt
    ```